"""
Comparison Service — overlay multiple group plots.

Pure Python + matplotlib, no Qt dependencies. Loads 2+ group .npz files
and produces publication-quality comparison plots with mean +/- SEM ribbons.

Usage:
    from core.services.comparison_service import compare_groups
    fig = compare_groups(group_paths, metric_keys=["if", "ti", "amp_insp"])
    fig.savefig("comparison.png", dpi=300)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Default color palette for groups (colorblind-friendly)
GROUP_COLORS = [
    "#2196F3",  # Blue
    "#E91E63",  # Pink
    "#4CAF50",  # Green
    "#FF9800",  # Orange
    "#9C27B0",  # Purple
    "#00BCD4",  # Cyan
    "#795548",  # Brown
    "#607D8B",  # Blue Gray
]

# Human-readable metric labels
METRIC_LABELS = {
    "if": "Inst. Frequency (Hz)",
    "ti": "Insp. Time (s)",
    "te": "Exp. Time (s)",
    "amp_insp": "Insp. Amplitude",
    "amp_exp": "Exp. Amplitude",
    "area_insp": "Insp. Area",
    "area_exp": "Exp. Area",
    "vent_proxy": "Ventilation Proxy",
}


def compare_groups(
    group_paths: List[Path],
    metric_keys: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    show_individual: bool = False,
    alpha_ribbon: float = 0.25,
    alpha_individual: float = 0.15,
    stim_window: Optional[Tuple[float, float]] = None,
):
    """Load group .npz files and create comparison overlay plot.

    Args:
        group_paths: Paths to group .npz files (2+ groups).
        metric_keys: Which metrics to plot (default: all 8).
        title: Figure title (default: auto-generated from group names).
        figsize: Figure size in inches (default: auto-scaled).
        show_individual: If True, show individual experiment traces.
        alpha_ribbon: Opacity of the SEM ribbon (0-1).
        alpha_individual: Opacity of individual traces.
        stim_window: Optional (start, end) to draw stim highlight.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    from core.services.grouping_service import load_group

    # Load all groups
    groups = []
    for gp in group_paths:
        groups.append(load_group(gp))

    if not groups:
        raise ValueError("No groups to compare")

    # Determine metrics to plot
    if metric_keys is None:
        # Use intersection of all groups' available metrics
        available = set(groups[0]["metrics"].keys())
        for g in groups[1:]:
            available &= set(g["metrics"].keys())
        metric_keys = sorted(available)

    if not metric_keys:
        raise ValueError("No common metrics found across groups")

    n_metrics = len(metric_keys)
    if figsize is None:
        figsize = (12, 2.5 * n_metrics)

    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True, squeeze=False)
    axes = axes.flatten()

    # Auto-detect stim window from first group's stim_duration or metadata
    if stim_window is None:
        stim_dur = groups[0].get("stim_duration", 0.0)
        if stim_dur > 0:
            stim_window = (0.0, stim_dur)
        else:
            meta = groups[0].get("metadata", {})
            if "stim_duration" in meta:
                stim_window = (0.0, float(meta["stim_duration"]))
            else:
                # Default: assume 0-15s stim if time range includes 0
                t0 = groups[0]["t_common"]
                if t0.min() < 0 and t0.max() > 15:
                    stim_window = (0.0, 15.0)

    for i, key in enumerate(metric_keys):
        ax = axes[i]

        for j, group in enumerate(groups):
            color = GROUP_COLORS[j % len(GROUP_COLORS)]
            label = f"{group['group_name']} (n={group['n_experiments']})"
            t = group["t_common"]

            if key not in group["metrics"]:
                continue

            m = group["metrics"][key]
            mean = m["mean"]
            sem = m["sem"]

            # Plot mean line
            ax.plot(t, mean, color=color, linewidth=1.5, label=label, zorder=3)

            # Plot SEM ribbon
            upper = mean + sem
            lower = mean - sem
            ax.fill_between(t, lower, upper, color=color, alpha=alpha_ribbon, zorder=2)

            # Plot individual traces if requested
            if show_individual and "individual" in m:
                individual = m["individual"]  # (n_timepoints, n_experiments)
                for k in range(individual.shape[1]):
                    ax.plot(t, individual[:, k], color=color, alpha=alpha_individual,
                            linewidth=0.5, zorder=1)

        # Stim window highlight (on every subplot)
        if stim_window is not None:
            ax.axvspan(stim_window[0], stim_window[1], color="#4682B4", alpha=0.15, zorder=0)
            ax.axvline(stim_window[0], color="#4682B4", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.axvline(stim_window[1], color="#4682B4", linewidth=0.8, linestyle="--", alpha=0.6)
            if i == 0:
                ax.text(
                    (stim_window[0] + stim_window[1]) / 2, ax.get_ylim()[1],
                    "Laser", ha="center", va="bottom", fontsize=8,
                    color="#4682B4", fontweight="bold",
                )

        ax.set_ylabel(METRIC_LABELS.get(key, key), fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if i == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    # X-axis label on bottom subplot
    axes[-1].set_xlabel("Time (s)", fontsize=11)

    # Title
    if title is None:
        group_names = [g["group_name"] for g in groups]
        title = " vs ".join(group_names)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def compare_groups_summary_bars(
    group_paths: List[Path],
    metric_keys: Optional[List[str]] = None,
    windows: Optional[List[Tuple[str, float, float]]] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    """Create bar chart comparing group means across time windows.

    Args:
        group_paths: Paths to group .npz files.
        metric_keys: Which metrics to include (default: ["if", "amp_insp"]).
        windows: Time windows as (label, t_start, t_end) tuples.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    from core.services.grouping_service import load_group

    if metric_keys is None:
        metric_keys = ["if", "amp_insp"]

    if windows is None:
        windows = [
            ("Baseline", -10.0, 0.0),
            ("Stim 0-5s", 0.0, 5.0),
            ("Stim 5-10s", 5.0, 10.0),
            ("Stim 10-15s", 10.0, 15.0),
            ("Post 15-25s", 15.0, 25.0),
        ]

    groups = [load_group(gp) for gp in group_paths]
    n_groups = len(groups)
    n_windows = len(windows)

    fig, axes = plt.subplots(len(metric_keys), 1, figsize=figsize or (10, 3.5 * len(metric_keys)))
    if len(metric_keys) == 1:
        axes = [axes]

    bar_width = 0.8 / n_groups
    x = np.arange(n_windows)

    for mi, key in enumerate(metric_keys):
        ax = axes[mi]
        for gi, group in enumerate(groups):
            if key not in group["metrics"]:
                continue

            t = group["t_common"]
            mean = group["metrics"][key]["mean"]
            sem = group["metrics"][key]["sem"]

            window_means = []
            window_sems = []
            for _, t_start, t_end in windows:
                mask = (t >= t_start) & (t < t_end)
                if mask.sum() > 0:
                    window_means.append(np.nanmean(mean[mask]))
                    # SEM of the window mean (propagate SEM)
                    window_sems.append(np.nanmean(sem[mask]))
                else:
                    window_means.append(np.nan)
                    window_sems.append(np.nan)

            color = GROUP_COLORS[gi % len(GROUP_COLORS)]
            offset = (gi - n_groups / 2 + 0.5) * bar_width
            ax.bar(x + offset, window_means, bar_width, yerr=window_sems,
                   color=color, alpha=0.8, label=group["group_name"],
                   capsize=3, edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([w[0] for w in windows], rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(METRIC_LABELS.get(key, key), fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if mi == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Group Comparison — Window Means", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig
