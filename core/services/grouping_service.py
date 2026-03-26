"""
Grouping Service — create group .npz from selected experiments.

Pure Python, no Qt dependencies. Loads y2 continuous metrics from .pmx files,
aligns to a common time grid, computes mean +/- SEM, and saves a group .npz.

Usage:
    from core.services.grouping_service import create_group
    result = create_group(pmx_paths, "Males 10mW", output_dir)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import numpy as np

from core.domain.analysis.models import GroupResult


# The 8 core continuous metrics saved by batch analysis
Y2_METRIC_KEYS = ["if", "ti", "te", "amp_insp", "amp_exp", "area_insp", "area_exp", "vent_proxy"]


def _load_y2_from_pmx(pmx_path: Path) -> Dict[str, Any]:
    """Load continuous y2 metrics and metadata from a .pmx file.

    Returns dict with keys: t, sr_hz, analyze_chan, y2 (dict of metric -> 2D array),
    stim_spans (list of (onset, offset) tuples), animal_id, original_file.
    """
    data = np.load(str(pmx_path), allow_pickle=True)

    t = data["t"] if "t" in data else None

    # Try loading embedded time array; fall back to reconstructing from sr_hz
    if t is None and "sr_hz" in data:
        sr_hz = float(data["sr_hz"])
        # Need to know length — get from first y2 array
        if "y2_continuous_sweep_indices" in data:
            idx = data["y2_continuous_sweep_indices"]
            keys_json = str(data["y2_continuous_metric_keys_json"])
            metric_keys = json.loads(keys_json)
            first_key = f"y2c_{metric_keys[0]}_sweep_{idx[0]}"
            if first_key in data:
                n = len(data[first_key])
                t = np.arange(n) / sr_hz

    if t is None:
        raise ValueError(f"No time array found in {pmx_path.name}")

    # Load y2 continuous metrics
    y2 = {}  # metric_key -> list of 1D arrays (one per sweep)
    if "y2_continuous_sweep_indices" not in data:
        available_keys = [k for k in data.files if 'y2' in k.lower()]
        raise ValueError(
            f"No y2 continuous metrics in {pmx_path.name} — re-run batch analysis.\n"
            f"Available y2-related keys: {available_keys}\n"
            f"Total keys: {len(data.files)}, has t={('t' in data.files)}, has sr_hz={('sr_hz' in data.files)}"
        )

    y2c_indices = data["y2_continuous_sweep_indices"]
    y2c_keys = json.loads(str(data["y2_continuous_metric_keys_json"]))

    for key in y2c_keys:
        sweep_arrays = []
        for s in y2c_indices:
            arr_key = f"y2c_{key}_sweep_{s}"
            if arr_key in data:
                sweep_arrays.append(data[arr_key])
        if sweep_arrays:
            y2[key] = sweep_arrays

    # Stim spans (for time normalization)
    stim_spans = []
    if "stim_spans_sweep_indices" in data:
        for s in data["stim_spans_sweep_indices"]:
            spans_json = str(data[f"stim_spans_sweep_{s}_json"])
            spans = json.loads(spans_json)
            stim_spans.extend([tuple(sp) for sp in spans])
            break  # Use first sweep's spans

    # Metadata
    original_file = str(data.get("original_file_path", ""))
    analyze_chan = str(data.get("analyze_chan", ""))

    data.close()

    return {
        "t": t,
        "y2": y2,
        "stim_spans": stim_spans,
        "original_file": original_file,
        "analyze_chan": analyze_chan,
        "pmx_path": str(pmx_path),
    }


def _mean_across_sweeps(sweep_arrays: List[np.ndarray]) -> np.ndarray:
    """Average multiple sweep arrays into a single 1D array."""
    if len(sweep_arrays) == 1:
        return sweep_arrays[0]
    stacked = np.column_stack(sweep_arrays)
    return np.nanmean(stacked, axis=1)


def _calc_mean_sem(data_2d: np.ndarray):
    """Calculate mean and SEM across columns (experiments) for each time point.

    Args:
        data_2d: shape (n_timepoints, n_experiments)

    Returns:
        (mean_1d, sem_1d) arrays
    """
    n = np.sum(np.isfinite(data_2d), axis=1)
    mean = np.full(data_2d.shape[0], np.nan)
    sem = np.full(data_2d.shape[0], np.nan)

    valid = n > 0
    if valid.any():
        mean[valid] = np.nanmean(data_2d[valid, :], axis=1)
        has_sem = n >= 2
        if has_sem.any():
            std = np.nanstd(data_2d[has_sem, :], axis=1, ddof=1)
            sem[has_sem] = std / np.sqrt(n[has_sem])

    return mean, sem


def create_group(
    pmx_paths: List[Path],
    group_name: str,
    output_dir: Path,
    metadata: Optional[Dict[str, Any]] = None,
    normalize_to_stim: bool = True,
    animal_ids: Optional[List[str]] = None,
) -> "GroupResult":
    """Create a group .npz from multiple .pmx experiment files.

    Args:
        pmx_paths: Paths to .pmx files (one per experiment).
        group_name: Human-readable group name (e.g., "Males 10mW").
        output_dir: Directory to save the group .npz file.
        metadata: Optional metadata dict (strain, stim_type, power, sex, ...).
        normalize_to_stim: If True, align time so t=0 is first stim onset.
        animal_ids: Optional list of animal IDs (same order as pmx_paths).

    Returns:
        GroupResult with group_path set on success.
    """
    result = GroupResult(group_name=group_name, metadata=metadata or {})

    try:
        # 1. Load y2 data from each .pmx
        experiments = []
        for i, pmx in enumerate(pmx_paths):
            if not pmx.exists():
                result.error = f"File not found: {pmx}"
                return result
            exp = _load_y2_from_pmx(pmx)
            if animal_ids and i < len(animal_ids):
                exp["animal_id"] = animal_ids[i]
            else:
                exp["animal_id"] = pmx.stem.split(".")[-1]  # Extract from filename
            experiments.append(exp)

        n_exp = len(experiments)
        result.n_experiments = n_exp
        result.source_files = [e["pmx_path"] for e in experiments]
        result.source_animal_ids = [e["animal_id"] for e in experiments]

        # 2. Compute per-experiment mean across sweeps for each metric
        # Also collect time arrays for common grid
        exp_means = []  # list of {metric_key: 1D array}
        exp_times = []

        for exp in experiments:
            t_exp = exp["t"]

            # Time normalization: shift so t=0 is first stim onset
            if normalize_to_stim and exp["stim_spans"]:
                t0 = exp["stim_spans"][0][0]
                t_exp = t_exp - t0

            exp_times.append(t_exp)

            means = {}
            for key in Y2_METRIC_KEYS:
                if key in exp["y2"]:
                    means[key] = _mean_across_sweeps(exp["y2"][key])
                else:
                    means[key] = np.full(len(t_exp), np.nan)
            exp_means.append(means)

        # 3. Build common time grid
        all_t_mins = [t.min() for t in exp_times]
        all_t_maxs = [t.max() for t in exp_times]
        all_steps = []
        for t_arr in exp_times:
            if len(t_arr) > 1:
                all_steps.append(np.median(np.diff(t_arr)))

        t_common_step = np.median(all_steps) if all_steps else 0.001
        t_common = np.arange(min(all_t_mins), max(all_t_maxs) + t_common_step / 2, t_common_step)

        # 4. Interpolate each experiment to common grid and compute mean/SEM
        group_data = {
            "version": "group_v1",
            "group_name": group_name,
            "source_files_json": json.dumps(result.source_files),
            "source_animal_ids_json": json.dumps(result.source_animal_ids),
            "t_common": t_common,
            "n_experiments": n_exp,
            "metadata_json": json.dumps(metadata or {}),
            "metric_keys_json": json.dumps(Y2_METRIC_KEYS),
        }

        from scipy.interpolate import interp1d

        for key in Y2_METRIC_KEYS:
            # Stack all experiments for this metric
            aligned = np.full((len(t_common), n_exp), np.nan)

            for j, (t_exp, means) in enumerate(zip(exp_times, exp_means)):
                y_exp = means.get(key, np.full(len(t_exp), np.nan))
                if len(t_exp) != len(t_common) or not np.allclose(t_exp, t_common, rtol=0.01):
                    fn = interp1d(t_exp, y_exp, kind="linear", bounds_error=False, fill_value=np.nan)
                    aligned[:, j] = fn(t_common)
                else:
                    aligned[:, j] = y_exp

            mean, sem = _calc_mean_sem(aligned)
            group_data[f"y2_{key}_mean"] = mean
            group_data[f"y2_{key}_sem"] = sem
            # Also save individual experiment means for potential re-analysis
            group_data[f"y2_{key}_individual"] = aligned

        result.metric_keys = list(Y2_METRIC_KEYS)

        # 5. Save group .npz
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = group_name.replace(" ", "_").replace("/", "_")
        group_path = output_dir / f"group_{safe_name}.npz"
        np.savez_compressed(str(group_path), **group_data)
        result.group_path = group_path

        return result

    except Exception as e:
        import traceback
        result.error = f"{type(e).__name__}: {e}"
        print(f"[grouping] Error: {traceback.format_exc()}")
        return result


def load_group(group_path: Path) -> Dict[str, Any]:
    """Load a group .npz file and return structured data.

    Returns:
        Dict with keys: group_name, t_common, metrics (dict of key -> {mean, sem}),
        source_files, source_animal_ids, n_experiments, metadata.
    """
    data = np.load(str(group_path), allow_pickle=True)

    group_name = str(data.get("group_name", "Unknown"))
    t_common = data["t_common"]
    n_exp = int(data.get("n_experiments", 0))
    metric_keys = json.loads(str(data.get("metric_keys_json", "[]")))

    metrics = {}
    for key in metric_keys:
        mean_key = f"y2_{key}_mean"
        sem_key = f"y2_{key}_sem"
        if mean_key in data and sem_key in data:
            metrics[key] = {
                "mean": data[mean_key],
                "sem": data[sem_key],
            }
            # Individual traces if available
            ind_key = f"y2_{key}_individual"
            if ind_key in data:
                metrics[key]["individual"] = data[ind_key]

    source_files = json.loads(str(data.get("source_files_json", "[]")))
    source_animal_ids = json.loads(str(data.get("source_animal_ids_json", "[]")))
    metadata = json.loads(str(data.get("metadata_json", "{}")))

    data.close()

    return {
        "group_name": group_name,
        "t_common": t_common,
        "metrics": metrics,
        "source_files": source_files,
        "source_animal_ids": source_animal_ids,
        "n_experiments": n_exp,
        "metadata": metadata,
    }
