"""
File loading service — business logic for file loading operations.

Pure Python, no Qt imports. Handles state mutation, channel auto-detection,
file validation, and path resolution. Used by FileLoadViewModel.
"""

from pathlib import Path
from typing import Optional

from core.domain.file_loading.models import (
    FileLoadResult,
    MultiFileLoadResult,
    NpzLoadResult,
    ChannelAutoDetection,
)


class FileLoadService:
    """Business logic for file loading operations."""

    def reset_analysis_state(self, state) -> None:
        """
        Clear all analysis results and caches from AppState.

        This is the DRY replacement for the duplicated "clear peaks/sighs/breaths/
        omitted/caches" block that appeared in all 3 completion handlers.
        """
        # Reset peak results
        if not hasattr(state, "peaks_by_sweep"):
            state.peaks_by_sweep = {}
            state.breath_by_sweep = {}
        else:
            state.peaks_by_sweep.clear()
            state.sigh_by_sweep.clear()
            state.breath_by_sweep.clear()
            state.omitted_sweeps.clear()
            state.omitted_ranges.clear()

        # Reset derived analysis state
        for attr in ('all_peaks_by_sweep', 'all_breaths_by_sweep',
                     'sniff_regions_by_sweep', 'peak_metrics_by_sweep',
                     'current_peak_metrics_by_sweep'):
            d = getattr(state, attr, None)
            if isinstance(d, dict):
                d.clear()

        # Clear event annotations
        if hasattr(state, 'bout_annotations'):
            state.bout_annotations.clear()

    def reset_caches(self, mw_state_holder) -> None:
        """
        Clear MainWindow-level caches that live outside AppState.

        Note: Outlier render caches (metrics_by_sweep, onsets_by_sweep,
        global_outlier_stats) are owned by PlotManager — call
        plot_manager.clear_caches() separately.

        Args:
            mw_state_holder: Object with _export_metric_cache,
                             zscore_global_mean, zscore_global_std attributes.
        """
        mw_state_holder._export_metric_cache = {}
        mw_state_holder.zscore_global_mean = None
        mw_state_holder.zscore_global_std = None

    def apply_single_file_data(self, state, result: FileLoadResult) -> None:
        """
        Apply loaded single-file data to AppState.

        Sets in_path, file_info, sr_hz, sweeps, channel_names, t, sweep_idx.
        """
        n_sweeps = next(iter(result.sweeps_by_ch.values())).shape[1]

        state.in_path = result.source_path
        state.file_info = [{
            'path': result.source_path,
            'sweep_start': 0,
            'sweep_end': n_sweeps - 1,
            **result.file_metadata,
        }]
        state.sr_hz = result.sr_hz
        state.sweeps = result.sweeps_by_ch
        state.channel_names = result.channel_names
        state.t = result.time_array
        state.sweep_idx = 0

    def apply_multi_file_data(self, state, result: MultiFileLoadResult) -> None:
        """
        Apply loaded multi-file data to AppState.

        Sets in_path (first file), file_info, sr_hz, sweeps, channel_names, t, sweep_idx.
        """
        state.in_path = result.source_paths[0] if result.source_paths else None
        state.file_info = result.file_info
        state.sr_hz = result.sr_hz
        state.sweeps = result.sweeps_by_ch
        state.channel_names = result.channel_names
        state.t = result.time_array
        state.sweep_idx = 0

    def auto_detect_channels(self, sweeps: dict, ch_names: list[str]) -> ChannelAutoDetection:
        """
        Auto-detect stimulus and analysis channels.

        Wraps abf_io.auto_select_channels. Returns ChannelAutoDetection with
        stim_channel and analysis_channel (analysis only set when exactly one
        non-stim channel exists).
        """
        from core import abf_io
        auto_stim, auto_analysis = abf_io.auto_select_channels(sweeps, ch_names)
        return ChannelAutoDetection(
            stim_channel=auto_stim,
            analysis_channel=auto_analysis,
        )

    def validate_files_for_concatenation(self, paths: list[Path]) -> tuple[bool, list[str]]:
        """
        Validate files before multi-file concatenation.

        Returns:
            (valid, messages) — valid=False means errors, messages may contain warnings.
        """
        from core import abf_io
        return abf_io.validate_files_for_concatenation(paths)

    def clear_stim_state(self, state) -> None:
        """Clear stimulus-related state fields."""
        state.stim_chan = None
        state.stim_onsets_by_sweep.clear()
        state.stim_offsets_by_sweep.clear()
        state.stim_spans_by_sweep.clear()
        state.stim_metrics_by_sweep.clear()

    def resolve_npz_data_path(
        self,
        metadata: dict,
        master_file_list: Optional[list[dict]] = None,
    ) -> Optional[Path]:
        """
        Resolve an alternative data path for an NPZ file whose original data
        file has moved. Searches the project's master file list for a matching
        filename.

        Args:
            metadata: NPZ metadata dict (from get_npz_metadata).
            master_file_list: List of file task dicts with 'file_path' keys.

        Returns:
            Path to the relocated file, or None if not found or not needed.
        """
        original_path_str = metadata.get('original_file', '')
        if not original_path_str:
            return None

        stored_path = Path(original_path_str)
        if stored_path.exists():
            return None  # Original path still works

        if not master_file_list:
            return None

        original_filename = stored_path.name
        print(f"\n[Path Resolution] NPZ stored path not found: {stored_path}")
        print(f"  Searching project file list for: {original_filename}")

        for task in master_file_list:
            task_path = task.get('file_path', '')
            if task_path and Path(task_path).name == original_filename:
                candidate = Path(task_path)
                if candidate.exists():
                    print(f"  Found in project: {candidate}")

                    # Analyze the difference for debugging
                    stored_parts = stored_path.parts
                    project_parts = candidate.parts
                    if stored_parts != project_parts:
                        for i, (s, p) in enumerate(zip(stored_parts, project_parts)):
                            if s != p:
                                print(f"  Path diverges at part {i}: '{s}' vs '{p}'")
                                break
                        if len(stored_parts) != len(project_parts):
                            print(f"  Path depth differs: {len(stored_parts)} vs {len(project_parts)} parts")

                    return candidate

        print(f"  Not found in project file list")
        return None

    def build_multi_file_summary(
        self,
        file_info: list[dict],
        sr_hz: float,
        sweeps_by_ch: dict,
        auto: ChannelAutoDetection,
    ) -> str:
        """
        Build a summary message string for multi-file loading result.

        Returns a message suitable for display in an info dialog.
        """
        total_sweeps = next(iter(sweeps_by_ch.values())).shape[1]

        file_lines = []
        for i, info in enumerate(file_info):
            line = f"  {i+1}. {info['path'].name}: sweeps {info['sweep_start']}-{info['sweep_end']}"
            if info.get('padded', False):
                orig_dur = info['original_samples'] / sr_hz
                padded_dur = info['padded_samples'] / sr_hz
                line += f" (padded: {orig_dur:.2f}s \u2192 {padded_dur:.2f}s)"
            file_lines.append(line)

        file_summary = "\n".join(file_lines)
        padded_count = sum(1 for info in file_info if info.get('padded', False))

        n_files = len(file_info)
        message = f"Loaded {n_files} files with {total_sweeps} total sweeps:\n\n{file_summary}"
        if padded_count > 0:
            message += f"\n\nNote: {padded_count} file(s) had different sweep lengths and were padded with NaN values."

        if auto.stim_channel:
            message += f"\n\nAuto-detected stimulus channel: {auto.stim_channel}"
            if auto.analysis_channel:
                message += f"\nAuto-selected analysis channel: {auto.analysis_channel}"

        return message
