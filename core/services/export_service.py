"""
ExportService — pure Python export computation functions.

No Qt dependencies. Extracted from ExportManager (export/export_manager.py)
as part of the MVVM refactoring (Step 4E).

All functions take state/config as parameters — no self.window refs.
ExportManager delegates to these; they can also be used headlessly.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from core import metrics


# ── String Utilities ────────────────────────────────────────────────


def sanitize_token(s: str) -> str:
    """Clean a string for use in filenames — strip, replace spaces, remove specials."""
    if not s:
        return ""
    s = s.strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    s = re.sub(r"_+", "_", s)
    s = re.sub(r"-+", "-", s)
    return s


# ── Metric Key Ordering ────────────────────────────────────────────


def metric_keys_in_order() -> List[str]:
    """Return metric keys in the UI order (from metrics.METRIC_SPECS)."""
    return [key for (_, key) in metrics.METRIC_SPECS]


# ── Statistics ──────────────────────────────────────────────────────


def nanmean_sem(X, axis=0):
    """Robust mean/SEM that avoids NumPy warnings with 0 or 1 finite values.

    Args:
        X: Array-like input
        axis: Axis to reduce along

    Returns:
        (mean, sem) arrays with NaN where insufficient data
    """
    A = np.asarray(X, dtype=float)
    if A.size == 0:
        return np.nan, np.nan

    A0 = np.moveaxis(A, axis, 0)
    tail = int(np.prod(A0.shape[1:])) or 1
    A2 = A0.reshape(A0.shape[0], tail)

    finite = np.isfinite(A2)
    n = finite.sum(axis=0)

    mean = np.full((tail,), np.nan, dtype=float)
    sem = np.full((tail,), np.nan, dtype=float)

    msk_mean = n > 0
    if np.any(msk_mean):
        mean[msk_mean] = np.nanmean(A2[:, msk_mean], axis=0)

    msk_sem = n >= 2
    if np.any(msk_sem):
        std = np.nanstd(A2[:, msk_sem], axis=0, ddof=1)
        sem[msk_sem] = std / np.sqrt(n[msk_sem])

    mean = mean.reshape(A0.shape[1:])
    sem = sem.reshape(A0.shape[1:])
    if mean.ndim > 1:
        mean = np.moveaxis(mean, 0, axis)
        sem = np.moveaxis(sem, 0, axis)

    return mean, sem


def mean_sem_1d(arr: np.ndarray) -> Tuple[float, float]:
    """Finite-only mean and SEM (ddof=1) for a 1D array.

    Returns (mean, sem). If no finite values → (nan, nan). If 1 → (mean, nan).
    """
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    n = int(finite.sum())
    if n == 0:
        return (np.nan, np.nan)
    vals = arr[finite]
    m = float(np.mean(vals))
    if n >= 2:
        s = float(np.std(vals, ddof=1))
        sem = s / np.sqrt(n)
    else:
        sem = np.nan
    return (m, sem)


# ── Breath Data Validation ─────────────────────────────────────────


def validate_breath_data(state) -> Tuple[bool, List[str]]:
    """Validate breath data before export.

    Checks for missing breath events, onset/offset mismatches,
    out-of-bounds indices, and overlapping breaths.

    Args:
        state: AppState with peaks_by_sweep, breath_by_sweep, t

    Returns:
        (is_valid, issues_list) where is_valid=True if no critical issues
    """
    issues = []
    warnings = []

    if state.t is None or len(state.t) == 0:
        issues.append("No time array available")
        return False, issues

    max_idx = len(state.t) - 1

    for sweep_idx in state.peaks_by_sweep.keys():
        breath_data = state.breath_by_sweep.get(sweep_idx, {})
        if not breath_data:
            continue

        onsets = breath_data.get('onsets', np.array([]))
        offsets = breath_data.get('offsets', np.array([]))

        onsets = np.asarray(onsets, dtype=int) if len(onsets) > 0 else np.array([], dtype=int)
        offsets = np.asarray(offsets, dtype=int) if len(offsets) > 0 else np.array([], dtype=int)

        if len(onsets) > 0 and len(offsets) > 0:
            if len(onsets) != len(offsets):
                warnings.append(f"Sweep {sweep_idx}: onset/offset count mismatch ({len(onsets)} vs {len(offsets)})")

        if len(onsets) > 0:
            bad_onsets = np.sum((onsets < 0) | (onsets > max_idx))
            if bad_onsets > 0:
                issues.append(f"Sweep {sweep_idx}: {bad_onsets} onset indices out of bounds")

        if len(offsets) > 0:
            bad_offsets = np.sum((offsets < 0) | (offsets > max_idx))
            if bad_offsets > 0:
                issues.append(f"Sweep {sweep_idx}: {bad_offsets} offset indices out of bounds")

        if len(onsets) > 1 and len(offsets) > 0:
            for i in range(min(len(onsets) - 1, len(offsets))):
                if onsets[i + 1] <= offsets[i]:
                    warnings.append(f"Sweep {sweep_idx}, breath {i}: possible overlap with next breath")
                    break

    is_valid = len(issues) == 0
    return is_valid, issues + warnings


# ── Sniffing Detection ──────────────────────────────────────────────


def is_breath_sniffing(state, sweep_idx: int, breath_idx: int, onsets: np.ndarray) -> bool:
    """Check if a breath is in a sniffing region based on its midpoint time.

    Args:
        state: AppState with sniff_regions_by_sweep, t
        sweep_idx: Sweep index
        breath_idx: Breath index (0-based)
        onsets: Array of onset sample indices

    Returns:
        True if breath midpoint falls in a sniffing region
    """
    sniff_regions = state.sniff_regions_by_sweep.get(sweep_idx, [])
    if not sniff_regions or breath_idx >= len(onsets) - 1:
        return False

    t_start = state.t[onsets[breath_idx]]
    t_end = state.t[onsets[breath_idx + 1]]
    t_mid = (t_start + t_end) / 2.0

    for (region_start, region_end) in sniff_regions:
        if region_start <= t_mid <= region_end:
            return True
    return False


# ── Sigh Indices ────────────────────────────────────────────────────


def sigh_sample_indices(state, sweep_idx: int, peaks: Optional[np.ndarray] = None) -> Set[int]:
    """Return sample indices of sigh-marked peaks on a sweep.

    Handles multiple storage patterns: sample indices, peak-list indices,
    or time values (seconds).

    Args:
        state: AppState with sigh_by_sweep or similar, t
        sweep_idx: Sweep index
        peaks: Optional peak sample indices for index mapping

    Returns:
        Set of sample indices (into state.t / y)
    """
    N = len(state.t)

    candidates = None
    for name in ("sighs_by_sweep", "sigh_indices_by_sweep", "sigh_peaks_by_sweep",
                 "sigh_mask_by_sweep", "sigh_by_sweep"):
        if hasattr(state, name):
            candidates = getattr(state, name).get(sweep_idx, None)
            if candidates is not None:
                break

    if candidates is None:
        return set()

    arr = np.asarray(list(candidates))
    if arr.size == 0:
        return set()

    out: Set[int] = set()

    if arr.dtype.kind in "iu":
        arr = arr.astype(int)
        if peaks is not None and arr.size and arr.max(initial=-1) < len(peaks):
            for idx in arr:
                if 0 <= idx < len(peaks):
                    i = int(peaks[idx])
                    if 0 <= i < N:
                        out.add(i)
        else:
            for i in arr:
                if 0 <= i < N:
                    out.add(int(i))
        return out

    if arr.dtype.kind in "f":
        t = state.t
        for val in arr:
            try:
                i = int(np.clip(np.searchsorted(t, float(val)), 0, N - 1))
                out.add(i)
            except Exception:
                pass
        return out

    for v in arr:
        try:
            i = int(v)
            if 0 <= i < N:
                out.add(i)
        except Exception:
            try:
                i = int(np.clip(np.searchsorted(state.t, float(v)), 0, N - 1))
                out.add(i)
            except Exception:
                pass
    return out


# ── Stim Masks ──────────────────────────────────────────────────────


def get_stim_masks(state, sweep_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (baseline_mask, stim_mask, post_mask) boolean arrays for a sweep.

    Args:
        state: AppState with t, stim_chan, stim_spans_by_sweep

    Returns:
        (baseline_mask, stim_mask, post_mask) — each boolean array over state.t
    """
    t = state.t
    spans = state.stim_spans_by_sweep.get(sweep_idx, []) if state.stim_chan else []

    if not spans:
        B = np.ones_like(t, dtype=bool)
        Z = np.zeros_like(t, dtype=bool)
        return B, Z, Z

    starts = np.array([a for (a, _) in spans], dtype=float)
    ends = np.array([b for (_, b) in spans], dtype=float)
    t0 = np.min(starts)
    t1 = np.max(ends)

    stim_mask = np.zeros_like(t, dtype=bool)
    for (a, b) in spans:
        stim_mask |= (t >= a) & (t <= b)

    baseline_mask = t < t0
    post_mask = t > t1
    return baseline_mask, stim_mask, post_mask


# ── Metric Trace Computation ───────────────────────────────────────


def compute_metric_trace(
    state,
    key: str,
    t: np.ndarray,
    y: np.ndarray,
    sr_hz: float,
    peaks: np.ndarray,
    breaths: Dict,
    sweep: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Compute a single metric trace for a sweep.

    Handles GMM probability injection for sniff_conf/eupnea_conf,
    and ECG result injection for hr/rr_interval.

    Args:
        state: AppState (for GMM probs, ECG results)
        key: Metric key (e.g. "if", "ti", "amp_insp")
        t, y: Time and signal arrays
        sr_hz: Sample rate
        peaks: Peak sample indices
        breaths: Dict with onsets, offsets, expmins, expoffs
        sweep: Optional sweep index for GMM/ECG context

    Returns:
        Metric trace array (same length as t), or None
    """
    fn = metrics.METRICS[key]
    on = breaths.get("onsets") if breaths else None
    off = breaths.get("offsets") if breaths else None
    exm = breaths.get("expmins") if breaths else None
    exo = breaths.get("expoffs") if breaths else None

    # Inject GMM probabilities if needed
    gmm_probs = None
    if sweep is not None and hasattr(state, 'gmm_sniff_probabilities') and sweep in state.gmm_sniff_probabilities:
        gmm_probs = state.gmm_sniff_probabilities[sweep]
        metrics.set_gmm_probabilities(gmm_probs)

    # Inject ECG result if needed
    ecg_result = None
    if key in ('hr', 'rr_interval') and sweep is not None:
        ecg_results = getattr(state, 'ecg_results_by_sweep', {})
        ecg_result = ecg_results.get(sweep)
        metrics.set_ecg_result(ecg_result)

    try:
        result = fn(t, y, sr_hz, peaks, on, off, exm, exo)
    except TypeError:
        result = fn(t, y, sr_hz, peaks, on, off, exm)
    finally:
        if gmm_probs is not None:
            metrics.set_gmm_probabilities(None)
        if ecg_result is not None:
            metrics.set_ecg_result(None)

    return result
