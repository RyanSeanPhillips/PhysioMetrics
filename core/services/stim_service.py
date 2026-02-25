"""
StimService — headless stim detection across sweeps.

Pure Python (no Qt). Wraps core.stim.detect_threshold_crossings
with multi-sweep logic so batch analysis can detect stims without MainWindow.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional, Tuple


def detect_stims_for_sweep(
    stim_signal: np.ndarray,
    t: np.ndarray,
    thresh: float = 1.0,
) -> Dict[str, Any]:
    """Detect stimulation events in a single sweep.

    Args:
        stim_signal: 1-D stim channel signal
        t: Time array
        thresh: Voltage threshold for crossing detection

    Returns:
        Dict with 'onsets', 'offsets', 'spans', 'metrics'
    """
    from core.stim import detect_threshold_crossings

    on_idx, off_idx, spans, metrics = detect_threshold_crossings(
        stim_signal, t, thresh=thresh
    )
    return {
        "onsets": on_idx,
        "offsets": off_idx,
        "spans": spans,
        "metrics": metrics,
    }


def detect_stims_all_sweeps(
    stim_sweeps_2d: np.ndarray,
    t: np.ndarray,
    thresh: float = 1.0,
    skip_existing: Optional[Dict[int, Any]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Detect stims across all sweeps.

    Args:
        stim_sweeps_2d: (n_samples, n_sweeps) stim channel data
        t: Time array
        thresh: Detection threshold
        skip_existing: Optional dict of sweep indices to skip (already detected)

    Returns:
        Dict mapping sweep_idx to detection result dicts.
    """
    n_sweeps = stim_sweeps_2d.shape[1]
    results = {}

    for s in range(n_sweeps):
        if skip_existing and s in skip_existing:
            continue
        results[s] = detect_stims_for_sweep(stim_sweeps_2d[:, s], t, thresh)

    return results
