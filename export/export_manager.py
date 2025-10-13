"""
ExportManager - Handles all data export operations.

Extracted from main.py for better maintainability and easier customization
for different experiment types.
"""

import re
import csv
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt6.QtWidgets import QMessageBox, QFileDialog, QDialog, QProgressDialog, QApplication
from PyQt6.QtCore import Qt
from core import metrics
from dialogs import SaveMetaDialog


class ExportManager:
    """Manages all data export operations for the main window."""

    # metrics we won't include in CSV exports and PDFs
    _EXCLUDE_FOR_CSV = {"d1", "d2", "eupnic", "apnea", "regularity"}

    def __init__(self, main_window):
        """
        Initialize the ExportManager.

        Args:
            main_window: Reference to MainWindow instance
        """
        self.window = main_window

    def _load_save_dialog_history(self) -> dict:
        """Load autocomplete history for the Save Data dialog from QSettings."""
        history = {
            'strain': [],
            'virus': [],
            'location': [],
            'stim': [],
            'power': [],
            'animal': []
        }

        for key in history.keys():
            saved_list = self.window.settings.value(f"save_history/{key}", [])
            if isinstance(saved_list, str):
                # Single value, convert to list
                saved_list = [saved_list] if saved_list else []
            history[key] = saved_list if saved_list else []

        return history


    def _update_save_dialog_history(self, vals: dict):
        """Update autocomplete history with new values from the Save Data dialog."""
        max_history = 20  # Keep last 20 unique values for each field

        for key in ['strain', 'virus', 'location', 'stim', 'power', 'animal']:
            value = vals.get(key, '').strip()
            if not value:
                continue

            # Get current history
            current = self.window.settings.value(f"save_history/{key}", [])
            if isinstance(current, str):
                current = [current] if current else []
            elif not isinstance(current, list):
                current = []

            # Remove value if already exists (we'll re-add at front)
            if value in current:
                current.remove(value)

            # Add to front
            current.insert(0, value)

            # Trim to max_history
            current = current[:max_history]

            # Save back
            self.window.settings.setValue(f"save_history/{key}", current)


    def _sanitize_token(self, s: str) -> str:
        if not s:
            return ""
        s = s.strip()
        s = s.replace(" ", "_")
        # allow alnum, underscore, hyphen, dot
        s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
        # squeeze repeats of underscores/hyphens
        s = re.sub(r"_+", "_", s)
        s = re.sub(r"-+", "-", s)
        return s



    def _suggest_stim_string(self) -> str:
        """
        Build a stim name like '20Hz10s15ms' from detected stim metrics
        or '15msPulse' / '5sPulse' for single pulses.
        Rounding:
        - freq_hz -> nearest Hz
        - duration_s -> nearest second
        - pulse_width_s -> nearest millisecond (or nearest second if >1s)
        """
        st = self.window.state
        if not getattr(st, "stim_chan", None):
            return ""

        # Make sure current sweep has metrics if possible
        try:
            self.window._compute_stim_for_current_sweep()
        except Exception:
            pass

        # Prefer current sweep, else first one that has metrics
        m = st.stim_metrics_by_sweep.get(st.sweep_idx) if hasattr(st, "stim_metrics_by_sweep") else None
        if not m:
            for _s, _m in getattr(st, "stim_metrics_by_sweep", {}).items():
                if _m:
                    m = _m
                    break
        if not m:
            return ""

        freq_hz   = m.get("freq_hz", None)
        dur_s     = m.get("duration_s", None)
        pw_s      = m.get("pulse_width_s", None)

        def _round_int(x):  # safe int rounding
            if x is None:
                return None
            try:
                return int(round(float(x)))
            except Exception:
                return None

        # TRAIN: if freq present and > 0, use Freq + Duration + PW
        if freq_hz is not None and np.isfinite(freq_hz) and freq_hz > 0:
            F = _round_int(freq_hz)              # Hz
            D = _round_int(dur_s)                # s
            MS = _round_int((pw_s or 0) * 1000)  # ms

            parts = []
            if F is not None and F > 0:  parts.append(f"{F}Hz")
            if D is not None and D > 0:  parts.append(f"{D}s")
            if MS is not None and MS > 0: parts.append(f"{MS}ms")
            return "".join(parts) if parts else ""

        # SINGLE PULSE (no usable freq)
        # Prefer pulse width; if missing, fall back to duration
        width_s = pw_s if (pw_s is not None and np.isfinite(pw_s)) else dur_s
        if width_s is None:
            return ""

        if width_s < 1.0:
            ms = _round_int(width_s * 1000)
            if ms is None or ms <= 0:
                ms = 1
            return f"{ms}msPulse"
        else:
            secs = _round_int(width_s)
            if secs is None or secs <= 0:
                secs = 1
            return f"{secs}sPulse"




    def on_save_analyzed_clicked(self):
        """Save analyzed data to disk after prompting for location/name."""
        from PyQt6.QtWidgets import QProgressDialog, QApplication
        from PyQt6.QtCore import Qt

        # Create progress dialog
        progress = QProgressDialog("Preparing data export...", None, 0, 100, self.window)
        progress.setWindowTitle("PlethAnalysis")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)  # Show immediately
        progress.setValue(0)
        QApplication.processEvents()

        try:
            self._export_all_analyzed_data(preview_only=False, progress_dialog=progress)
        finally:
            progress.close()


    def on_view_summary_clicked(self):
        """Display interactive preview of the PDF summary without saving."""
        from PyQt6.QtWidgets import QProgressDialog, QApplication
        from PyQt6.QtCore import Qt

        # Create progress dialog
        progress = QProgressDialog("Generating summary preview...", None, 0, 100, self.window)
        progress.setWindowTitle("PlethAnalysis")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)  # Show immediately
        progress.setValue(0)
        QApplication.processEvents()

        try:
            self._export_all_analyzed_data(preview_only=True, progress_dialog=progress)
        finally:
            progress.close()


    # metrics we won't include in CSV exports and PDFs
    _EXCLUDE_FOR_CSV = {"d1", "d2", "eupnic", "apnea", "regularity"}


    def _metric_keys_in_order(self):
        """Return metric keys in the UI order (from metrics.METRIC_SPECS)."""
        return [key for (_, key) in metrics.METRIC_SPECS]


    def _compute_metric_trace(self, key, t, y, sr_hz, peaks, breaths):
        """
        Call the metric function, passing expoffs if it exists.
        Falls back to legacy signature when needed.
        """
        fn = metrics.METRICS[key]
        on  = breaths.get("onsets")   if breaths else None
        off = breaths.get("offsets")  if breaths else None
        exm = breaths.get("expmins")  if breaths else None
        exo = breaths.get("expoffs")  if breaths else None
        try:
            return fn(t, y, sr_hz, peaks, on, off, exm, exo)  # new signature
        except TypeError:
            return fn(t, y, sr_hz, peaks, on, off, exm)       # legacy signature


    def _get_stim_masks(self, s: int):
        """
        Build (baseline_mask, stim_mask, post_mask) boolean arrays over st.t for sweep s.
        Uses union of all stim spans for 'stim'.
        """
        st = self.window.state
        t = st.t
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
        if not spans:
            # no stim: whole trace is baseline; stim/post empty
            B = np.ones_like(t, dtype=bool)
            Z = np.zeros_like(t, dtype=bool)
            return B, Z, Z

        starts = np.array([a for (a, _) in spans], dtype=float)
        ends   = np.array([b for (_, b) in spans], dtype=float)
        t0 = np.min(starts)
        t1 = np.max(ends)

        stim_mask = np.zeros_like(t, dtype=bool)
        for (a, b) in spans:
            stim_mask |= (t >= a) & (t <= b)

        baseline_mask = t < t0
        post_mask     = t > t1
        return baseline_mask, stim_mask, post_mask

    #     """Return (nanmean, nansem) along axis; SEM uses ddof=1 where n>=2 else NaN."""
    #     with np.errstate(invalid="ignore"):
    #         # std with ddof=1; guard n<2


    def _nanmean_sem(self, X, axis=0):
        """
        Robust mean/SEM that avoids NumPy RuntimeWarnings when there are
        0 or 1 finite values along the chosen axis.
        """
        import numpy as np

        A = np.asarray(X, dtype=float)
        if A.size == 0:
            return np.nan, np.nan

        # Move the target axis to 0 so we can index rows easily
        A0 = np.moveaxis(A, axis, 0)      # shape: (N, ...rest...)
        # Collapse the rest to a single trailing dimension for simplicity
        tail = int(np.prod(A0.shape[1:])) or 1
        A2 = A0.reshape(A0.shape[0], tail)  # (N, T)

        finite = np.isfinite(A2)
        n = finite.sum(axis=0)              # (T,)

        mean = np.full((tail,), np.nan, dtype=float)
        sem  = np.full((tail,), np.nan, dtype=float)

        # rows/columns (along axis) with at least one finite value
        msk_mean = n > 0
        if np.any(msk_mean):
            mean[msk_mean] = np.nanmean(A2[:, msk_mean], axis=0)

        # rows/columns with at least two finite values (only here compute SEM)
        msk_sem = n >= 2
        if np.any(msk_sem):
            std = np.nanstd(A2[:, msk_sem], axis=0, ddof=1)
            sem[msk_sem] = std / np.sqrt(n[msk_sem])

        # reshape back to the original tail and move axis back
        mean = mean.reshape(A0.shape[1:])
        sem  = sem.reshape(A0.shape[1:])
        # move axis back to original position (only if result has enough dimensions)
        # After reduction along one axis, result has (A.ndim - 1) dimensions
        # We can only moveaxis if result is multi-dimensional
        if mean.ndim > 1:
            mean = np.moveaxis(mean, 0, axis)
            sem  = np.moveaxis(sem, 0, axis)

        return mean, sem








    def _export_all_analyzed_data(self, preview_only=False, progress_dialog=None):
        """
        Exports (or previews) analyzed data.

        If preview_only=True: Shows interactive PDF preview dialog without saving files.
        If preview_only=False: Prompts for location/name and exports files.

        Exports:
        1) <base>_bundle.npz
            - Downsampled processed trace (kept sweeps only)
            - Downsampled y2 metric traces (all keys)
            - Peaks/breaths/sighs per kept sweep
            - Stim spans per kept sweep
            - Meta

        2) <base>_means_by_time.csv
            - t (relative to global stim start if present)
            - For each metric: optional per-sweep traces, mean, sem
            - Then the same block normalized by per-sweep baseline window (_norm)
            - Then the same block normalized by pooled eupneic baseline (_norm_eupnea)

        3) <base>_breaths.csv
            - Wide layout:
                RAW blocks:  ALL | BASELINE | STIM | POST
                NORM blocks: ALL | BASELINE | STIM | POST (per-sweep time-based)
                NORM_EUPNEA blocks: ALL | BASELINE | STIM | POST (pooled eupneic)
            - Includes `is_sigh` column (1 if any sigh peak in that breath interval)

        4) <base>_events.csv
            - Event intervals: stimulus on/off, apnea episodes, eupnea regions
            - Columns: sweep, event_type, start_time, end_time, duration
            - Times are relative to global stim start if present

        5) <base>_summary.pdf (or preview dialog if preview_only=True)
        """
        import numpy as np, csv, json, time
        from PyQt6.QtWidgets import QApplication

        st = self.window.state
        if not getattr(st, "in_path", None):
            QMessageBox.information(self.window, "View Summary" if preview_only else "Save analyzed data", "Open an ABF first.")
            return
        if st.t is None or not st.analyze_chan or st.analyze_chan not in st.sweeps:
            QMessageBox.warning(self.window, "View Summary" if preview_only else "Save analyzed data", "No analyzed data available.")
            return

        # -------------------- Prompt for save location (skip if preview_only) --------------------
        if not preview_only:
            # --- Build an auto stim string from current sweep metrics, if available ---
            def _auto_stim_from_metrics() -> str:
                s = max(0, min(getattr(st, "sweep_idx", 0), self.window.navigation_manager._sweep_count()-1))
                m = st.stim_metrics_by_sweep.get(s, {}) if getattr(st, "stim_metrics_by_sweep", None) else {}
                if not m:
                    return ""
                def _ri(x):
                    try: return int(round(float(x)))
                    except Exception: return None

                f = _ri(m.get("freq_hz"))
                d = _ri(m.get("duration_s"))
                pw_s = m.get("pulse_width_s")
                n_pulses = m.get("n_pulses", None)

                if f is not None and d is not None and pw_s is not None:
                    pw_str = f"{_ri(pw_s)}s" if pw_s >= 1.0 else f"{_ri(pw_s * 1000)}ms"
                    return f"{f}Hz{d}s{pw_str}"

                if pw_s is not None and (n_pulses == 1 or f is None):
                    return f"{_ri(pw_s)}sPulse" if pw_s >= 1.0 else f"{_ri(pw_s * 1000)}msPulse"
                return ""

            abf_stem = st.in_path.stem
            chan = st.analyze_chan or ""
            auto_stim = _auto_stim_from_metrics()

            # --- Load autocomplete history ---
            history = self._load_save_dialog_history()

            # --- Name builder dialog (with auto stim suggestion) ---
            dlg = SaveMetaDialog(abf_name=abf_stem, channel=chan, parent=self.window, auto_stim=auto_stim, history=history)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return

            vals = dlg.values()

            # --- Update history with new values ---
            self._update_save_dialog_history(vals)
            suggested = self._sanitize_token(vals["preview"]) or "analysis"
            want_picker = bool(vals.get("choose_dir", False))

            target_exact = "Pleth_App_analysis"          # <- required folder name (lowercase 'a')
            target_lower = target_exact.lower()

            def _nearest_analysis_ancestor(p: Path) -> Path | None:
                # Return the closest ancestor (including self) named Pleth_App_analysis (case-insensitive)
                for cand in [p] + list(p.parents):
                    if cand.name.lower() == target_lower:
                        return cand
                return None

            if want_picker:
                # User chooses a folder; we keep your previous smart logic here.
                default_root = Path(self.window.settings.value("save_root", str(st.in_path.parent)))
                chosen = QFileDialog.getExistingDirectory(
                    self,
                    "Choose a folder (files may go into an existing 'Pleth_App_analysis' here)",
                    str(default_root)
                )
                if not chosen:
                    return
                chosen_path = Path(chosen)

                # 1) If chosen folder is inside an ancestor named Pleth_App_analysis → save THERE (the ancestor)
                anc = _nearest_analysis_ancestor(chosen_path)
                if anc is not None:
                    final_dir = anc
                else:
                    # 2) If the chosen folder already contains 'Pleth_App_analysis' or 'Pleth_App_Analysis' subfolder → use it
                    sub_exact   = chosen_path / "Pleth_App_analysis"
                    sub_variant = chosen_path / "Pleth_App_Analysis"
                    if sub_exact.is_dir():
                        final_dir = sub_exact
                    elif sub_variant.is_dir():
                        final_dir = sub_variant
                    else:
                        # 3) Otherwise, create Pleth_App_analysis directly under the chosen folder
                        final_dir = chosen_path / target_exact
                        try:
                            final_dir.mkdir(parents=True, exist_ok=True)
                        except Exception as e:
                            QMessageBox.critical(self.window, "Save analyzed data", f"Could not create folder:\n{final_dir}\n\n{e}")
                            return

                # Remember the last *picker* root only when the picker is used
                self.window.settings.setValue("save_root", str(chosen_path))

            else:
                # UNCHECKED: Always use the CURRENT ABF DIRECTORY (not a remembered one)
                base_root = st.in_path.parent if getattr(st, "in_path", None) else Path.cwd()
                final_dir = base_root / target_exact
                try:
                    final_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    QMessageBox.critical(self.window, "Save analyzed data", f"Could not create folder:\n{final_dir}\n\n{e}")
                    return
                # IMPORTANT: Do NOT overwrite 'save_root' here — we don't want to "remember" anything for the unchecked case.

            # Set base name + meta, then export
            self.window._save_dir = final_dir
            self.window._save_base = suggested
            self.window._save_meta = vals

            # Check for duplicate files before saving
            existing_files = []
            expected_suffixes = ["_bundle.npz", "_means_by_time.csv", "_breaths.csv", "_events.csv", "_summary.pdf"]
            for suffix in expected_suffixes:
                filepath = final_dir / (suggested + suffix)
                if filepath.exists():
                    existing_files.append(filepath.name)

            if existing_files:
                # Show warning dialog
                file_list = "\n  • ".join(existing_files)
                reply = QMessageBox.question(
                    self.window,
                    "Overwrite Existing Files?",
                    f"The following files already exist in:\n{final_dir}\n\n  • {file_list}\n\n"
                    f"Do you want to overwrite them?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No  # Default to No for safety
                )
                if reply == QMessageBox.StandardButton.No:
                    return  # User chose not to overwrite

            base_path = self.window._save_dir / self.window._save_base
            print(f"[save] base path set: {base_path}")
            try:
                self.window.statusbar.showMessage(f"Saving to: {base_path}", 4000)
            except Exception:
                pass

        # -------------------- knobs --------------------
        DS_TARGET_HZ    = 50.0
        CSV_FLUSH_EVERY = 2000
        INCLUDE_TRACES  = bool(getattr(self, "_csv_include_traces", True))
        NORM_BASELINE_WINDOW_S = float(getattr(self, "_norm_window_s", 10.0))
        EPS_BASE = 1e-12

        # -------------------- basics --------------------
        if progress_dialog:
            progress_dialog.setLabelText("Preparing data...")
            progress_dialog.setValue(5)
            QApplication.processEvents()

        any_ch    = next(iter(st.sweeps.values()))
        n_sweeps  = int(any_ch.shape[1])
        N         = int(len(st.t))

        kept_sweeps = [s for s in range(n_sweeps) if s not in getattr(st, "omitted_sweeps", set())]
        S = len(kept_sweeps)
        if S == 0:
            QMessageBox.warning(self.window, "Save analyzed data", "All sweeps are omitted. Nothing to save.")
            return

        # Downsample index used for NPZ + time CSV
        ds_step = max(1, int(round(float(st.sr_hz) / DS_TARGET_HZ))) if st.sr_hz else 1
        ds_idx  = np.arange(0, N, ds_step, dtype=int)
        M       = int(len(ds_idx))

        # Global stim zero and duration (union across KEPT sweeps)
        global_s0, global_s1 = None, None
        if st.stim_chan:
            for s in kept_sweeps:
                spans = st.stim_spans_by_sweep.get(s, [])
                if spans:
                    starts = [a for (a, _) in spans]
                    ends   = [b for (_, b) in spans]
                    m0 = float(min(starts)); m1 = float(max(ends))
                    global_s0 = m0 if global_s0 is None else min(global_s0, m0)
                    global_s1 = m1 if global_s1 is None else max(global_s1, m1)
        have_global_stim = (global_s0 is not None and global_s1 is not None)
        global_dur = (global_s1 - global_s0) if have_global_stim else None

        # Time for NPZ (raw) and for CSV (normalized to global_s0 if present)
        t_ds_raw = st.t[ds_idx]
        csv_t0   = (global_s0 if have_global_stim else 0.0)
        t_ds_csv = (st.t - csv_t0)[ds_idx]

        # -------------------- containers --------------------
        all_keys     = self._metric_keys_in_order()
        Y_proc_ds    = np.full((M, S), np.nan, dtype=float)
        y2_ds_by_key = {k: np.full((M, S), np.nan, dtype=float) for k in all_keys}

        peaks_by_sweep, on_by_sweep, off_by_sweep, exm_by_sweep, exo_by_sweep = [], [], [], [], []
        sigh_by_sweep = []

        # -------------------- fill per kept sweep --------------------
        for col, s in enumerate(kept_sweeps):
            y_proc = self.window._get_processed_for(st.analyze_chan, s)
            Y_proc_ds[:, col] = y_proc[ds_idx]

            pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
            br  = st.breath_by_sweep.get(s, None)
            if br is None and pks.size:
                # backfill breaths for this sweep so exports are consistent
                br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
                st.breath_by_sweep[s] = br
            if br is None:
                br = {
                    "onsets":  np.array([], dtype=int),
                    "offsets": np.array([], dtype=int),
                    "expmins": np.array([], dtype=int),
                    "expoffs": np.array([], dtype=int),
                }

            peaks_by_sweep.append(pks)
            on_by_sweep.append(np.asarray(br.get("onsets",  []), dtype=int))
            off_by_sweep.append(np.asarray(br.get("offsets", []), dtype=int))
            exm_by_sweep.append(np.asarray(br.get("expmins", []), dtype=int))
            exo_by_sweep.append(np.asarray(br.get("expoffs", []), dtype=int))

            # sighs for this sweep (peak indices)
            sighs = np.asarray(st.sigh_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
            # Keep only valid in-range indices
            sighs = sighs[(sighs >= 0) & (sighs < len(y_proc))]
            sigh_by_sweep.append(sighs)

            for k in all_keys:
                y2 = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, br)
                if y2 is not None and len(y2) == N:
                    y2_ds_by_key[k][:, col] = y2[ds_idx]

        if progress_dialog:
            progress_dialog.setLabelText("Computing metrics...")
            progress_dialog.setValue(15)
            QApplication.processEvents()

        # -------------------- Save files (skip if preview_only) --------------------
        if not preview_only:
            # -------------------- (1) NPZ bundle (downsampled) --------------------
            base     = self.window._save_dir / self.window._save_base
            npz_path = base.with_name(base.name + "_bundle.npz")
    
            # Pack stim spans in KEPT order (align with columns)
            stim_obj = np.empty(S, dtype=object)
            for col, s in enumerate(kept_sweeps):
                spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
                stim_obj[col] = np.array(spans, dtype=float).reshape(-1, 2) if spans else np.empty((0, 2), dtype=float)
    
            peaks_obj = np.array(peaks_by_sweep, dtype=object)
            on_obj    = np.array(on_by_sweep,  dtype=object)
            off_obj   = np.array(off_by_sweep, dtype=object)
            exm_obj   = np.array(exm_by_sweep, dtype=object)
            exo_obj   = np.array(exo_by_sweep, dtype=object)
            sigh_obj  = np.array(sigh_by_sweep, dtype=object)

            # Pack sniffing regions (aligned with kept_sweeps)
            sniff_obj = np.empty(S, dtype=object)
            for col, s in enumerate(kept_sweeps):
                regions = st.sniff_regions_by_sweep.get(s, [])
                # Convert to numpy array of shape (N_regions, 2) for start/end times
                sniff_obj[col] = np.array(regions, dtype=float).reshape(-1, 2) if regions else np.empty((0, 2), dtype=float)

            y2_kwargs_ds = {f"y2_{k}_ds": y2_ds_by_key[k] for k in all_keys}
    
            meta = {
                "analyze_channel": st.analyze_chan,
                "sr_hz": float(st.sr_hz),
                "n_sweeps_total": int(n_sweeps),
                "n_sweeps_kept": int(S),
                "kept_sweeps": [int(s) for s in kept_sweeps],                  # original indices
                "omitted_sweeps": sorted(int(x) for x in getattr(st, "omitted_sweeps", set())),
                "abf_path": str(getattr(st, "in_path", "")),
                "ui_meta": getattr(self, "_save_meta", {}),
                "excluded_for_csv": sorted(list(self._EXCLUDE_FOR_CSV)),
                "ds_target_hz": float(DS_TARGET_HZ),
                "ds_step": int(ds_step),
                "csv_time_zero": float(csv_t0),
                "csv_includes_traces": bool(INCLUDE_TRACES),
                "norm_window_s": float(NORM_BASELINE_WINDOW_S),
                # Filter settings (for future NPZ reopening)
                "use_low": bool(st.use_low),
                "use_high": bool(st.use_high),
                "use_mean_sub": bool(st.use_mean_sub),
                "use_invert": bool(st.use_invert),
                "low_hz": float(st.low_hz) if st.low_hz else None,
                "high_hz": float(st.high_hz) if st.high_hz else None,
                "mean_val": float(st.mean_val),
                "filter_order": int(self.window.filter_order),
                # Notch filter (if active)
                "notch_filter_lower": float(self.window.notch_filter_lower) if self.window.notch_filter_lower else None,
                "notch_filter_upper": float(self.window.notch_filter_upper) if self.window.notch_filter_upper else None,
                # Channel info (for reopening)
                "channel_names": list(st.channel_names) if st.channel_names else [],
                "stim_chan": str(st.stim_chan) if st.stim_chan else None,
                # Navigation state
                "window_dur_s": float(st.window_dur_s),
            }
    
            # Save enhanced NPZ bundle with timeseries data (for fast consolidation)
            # This will be updated after timeseries normalization is computed
            _npz_timeseries_data = {
                't_ds': t_ds_raw,
                'Y_proc_ds': Y_proc_ds,
                'peaks_by_sweep': peaks_obj,
                'onsets_by_sweep': on_obj,
                'offsets_by_sweep': off_obj,
                'expmins_by_sweep': exm_obj,
                'expoffs_by_sweep': exo_obj,
                'sigh_idx_by_sweep': sigh_obj,
                'sniff_regions_by_sweep': sniff_obj,
                'stim_spans_by_sweep': stim_obj,
                'meta_json': json.dumps(meta),
                **y2_kwargs_ds,
            }
    
            # -------------------- helpers for normalization (time CSV) --------------------
            def _per_sweep_baseline_for_time(A_ds: np.ndarray) -> np.ndarray:
                """
                A_ds: (M,S) downsampled metric matrix.
                Returns b[S]: mean over last NORM_BASELINE_WINDOW_S before 0; fallback to first W after 0.
                """
                b = np.full((A_ds.shape[1],), np.nan, dtype=float)
                mask_pre  = (t_ds_csv >= -NORM_BASELINE_WINDOW_S) & (t_ds_csv < 0.0)
                mask_post = (t_ds_csv >=  0.0) & (t_ds_csv <= NORM_BASELINE_WINDOW_S)
                for sidx in range(A_ds.shape[1]):
                    col = A_ds[:, sidx]
                    vals = col[mask_pre]
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        vals = col[mask_post]
                        vals = vals[np.isfinite(vals)]
                    if vals.size:
                        b[sidx] = float(np.mean(vals))
                return b
    
            def _normalize_matrix_by_baseline(A_ds: np.ndarray, b: np.ndarray) -> np.ndarray:
                out = np.full_like(A_ds, np.nan)
                for sidx in range(A_ds.shape[1]):
                    bs = b[sidx]
                    if np.isfinite(bs) and abs(bs) > EPS_BASE:
                        out[:, sidx] = A_ds[:, sidx] / bs
                return out
            
        
    
    
            # -------------------- (2) Per-time CSV (raw + normalized appended) --------------------
            csv_time_path = base.with_name(base.name + "_timeseries.csv")
            keys_for_csv  = [k for k in all_keys if k not in self._EXCLUDE_FOR_CSV]
    
            if progress_dialog:
                progress_dialog.setLabelText("Computing time-based normalization...")
                progress_dialog.setValue(25)
                QApplication.processEvents()

            # Build normalized stacks per metric (TIME-BASED, per-sweep)
            y2_ds_by_key_norm = {}
            baseline_by_key   = {}
            for k in keys_for_csv:
                b = _per_sweep_baseline_for_time(y2_ds_by_key[k])
                baseline_by_key[k] = b
                y2_ds_by_key_norm[k] = _normalize_matrix_by_baseline(y2_ds_by_key[k], b)

            if progress_dialog:
                progress_dialog.setLabelText("Computing eupnea-based normalization...")
                progress_dialog.setValue(35)
                QApplication.processEvents()

            # Build EUPNEA-BASED normalization (pooled across all sweeps)
            # Use efficient approach: extract from pre-computed matrices
            y2_ds_by_key_norm_eupnea = {}
            eupnea_baseline_by_key = {}

            # Get thresholds from UI
            eupnea_thresh = self.window.eupnea_freq_threshold  # Hz
            apnea_thresh = self.window._parse_float(self.window.ApneaThresh) or 0.5    # seconds

            # Pre-compute eupnea masks once per sweep
            print(f"[CSV-time] Pre-computing eupnea masks for {len(kept_sweeps)} sweeps...")
            eupnea_masks_csv = {}
            for s in kept_sweeps:
                y_proc = self.window._get_processed_for(st.analyze_chan, s)
                pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                br = st.breath_by_sweep.get(s, None)
                if br is None and pks.size:
                    br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
                    st.breath_by_sweep[s] = br
                if br is None:
                    continue

                on = np.asarray(br.get("onsets", []), dtype=int)
                off = np.asarray(br.get("offsets", []), dtype=int)
                expmins = np.asarray(br.get("expmins", []), dtype=int)
                expoffs = np.asarray(br.get("expoffs", []), dtype=int)

                if on.size >= 2:
                    # Get sniff regions for this sweep
                    sniff_regions = self.window.state.sniff_regions_by_sweep.get(s, [])
                    eupnea_mask = metrics.detect_eupnic_regions(
                        st.t, y_proc, st.sr_hz, pks, on, off, expmins, expoffs,
                        freq_threshold_hz=eupnea_thresh,
                        min_duration_sec=self.window.eupnea_min_duration,
                        sniff_regions=sniff_regions
                    )
                    eupnea_masks_csv[s] = eupnea_mask

            # Compute baselines by extracting from y2_ds_by_key matrices
            # OPTIMIZATION: Build index mapping once instead of calling argmin repeatedly
            print(f"[CSV-time] Computing eupnea baselines for {len(keys_for_csv)} metrics...")

            # Pre-compute mapping from downsampled indices to original indices
            t0 = float(global_s0) if have_global_stim else 0.0
            ds_to_orig_idx = np.zeros(len(t_ds_csv), dtype=int)
            for ds_idx in range(len(t_ds_csv)):
                t_target = t_ds_csv[ds_idx] + t0
                ds_to_orig_idx[ds_idx] = np.argmin(np.abs(st.t - t_target))

            # Identify baseline indices (t < 0)
            baseline_ds_mask = t_ds_csv < 0
            poststim_ds_mask = (t_ds_csv >= 0) & (t_ds_csv <= NORM_BASELINE_WINDOW_S)

            for k in keys_for_csv:
                Y_raw = y2_ds_by_key.get(k, None)
                if Y_raw is None or Y_raw.size == 0:
                    eupnea_baseline_by_key[k] = np.nan
                    y2_ds_by_key_norm_eupnea[k] = np.full_like(Y_raw, np.nan) if Y_raw is not None else None
                    continue

                pooled_vals = []

                # Collect eupneic baseline values from downsampled data
                for col_idx, s in enumerate(kept_sweeps):
                    eupnea_mask = eupnea_masks_csv.get(s, None)
                    if eupnea_mask is None:
                        continue

                    # Check baseline time points
                    for ds_idx in np.where(baseline_ds_mask)[0]:
                        i_orig = ds_to_orig_idx[ds_idx]

                        # Check if this point is eupneic
                        if i_orig < len(eupnea_mask) and eupnea_mask[i_orig] > 0:
                            val = Y_raw[ds_idx, col_idx]
                            if np.isfinite(val):
                                pooled_vals.append(val)

                # Fallback: include post-stim eupneic periods if insufficient
                if len(pooled_vals) < 10:
                    for col_idx, s in enumerate(kept_sweeps):
                        eupnea_mask = eupnea_masks_csv.get(s, None)
                        if eupnea_mask is None:
                            continue

                        for ds_idx in np.where(poststim_ds_mask)[0]:
                            i_orig = ds_to_orig_idx[ds_idx]

                            if i_orig < len(eupnea_mask) and eupnea_mask[i_orig] > 0:
                                val = Y_raw[ds_idx, col_idx]
                                if np.isfinite(val):
                                    pooled_vals.append(val)

                # Compute baseline and normalize
                eupnea_b = float(np.mean(pooled_vals)) if len(pooled_vals) > 0 else np.nan
                eupnea_baseline_by_key[k] = eupnea_b

                if np.isfinite(eupnea_b) and abs(eupnea_b) > EPS_BASE:
                    y2_ds_by_key_norm_eupnea[k] = Y_raw / eupnea_b
                else:
                    y2_ds_by_key_norm_eupnea[k] = np.full_like(Y_raw, np.nan)

            # headers: raw first, then time-based norm, then eupnea-based norm
            header = ["t"]
            for k in keys_for_csv:
                if INCLUDE_TRACES:
                    header += [f"{k}_s{j+1}" for j in range(S)]
                header += [f"{k}_mean", f"{k}_sem"]

            for k in keys_for_csv:
                if INCLUDE_TRACES:
                    header += [f"{k}_norm_s{j+1}" for j in range(S)]
                header += [f"{k}_norm_mean", f"{k}_norm_sem"]

            for k in keys_for_csv:
                if INCLUDE_TRACES:
                    header += [f"{k}_norm_eupnea_s{j+1}" for j in range(S)]
                header += [f"{k}_norm_eupnea_mean", f"{k}_norm_eupnea_sem"]
    
            if progress_dialog:
                progress_dialog.setLabelText("Writing time-series CSV...")
                progress_dialog.setValue(50)
                QApplication.processEvents()

            t_start = time.time()
            self.window.setCursor(Qt.CursorShape.WaitCursor)
            try:
                # Build DataFrame from existing numpy arrays (2-3× faster than row-by-row)
                import pandas as pd

                # Build columns in exact order to match header
                data = {}
                data['t'] = t_ds_csv

                # RAW block
                for k in keys_for_csv:
                    if INCLUDE_TRACES:
                        for j in range(S):
                            data[f'{k}_s{j+1}'] = y2_ds_by_key[k][:, j]
                    # Compute mean and SEM across sweeps
                    means, sems = self._nanmean_sem(y2_ds_by_key[k], axis=1)
                    data[f'{k}_mean'] = means
                    data[f'{k}_sem'] = sems

                # NORMALIZED block (time-based, per-sweep)
                for k in keys_for_csv:
                    if INCLUDE_TRACES:
                        for j in range(S):
                            data[f'{k}_norm_s{j+1}'] = y2_ds_by_key_norm[k][:, j]
                    means_n, sems_n = self._nanmean_sem(y2_ds_by_key_norm[k], axis=1)
                    data[f'{k}_norm_mean'] = means_n
                    data[f'{k}_norm_sem'] = sems_n

                # NORMALIZED block (eupnea-based, pooled)
                for k in keys_for_csv:
                    if INCLUDE_TRACES:
                        for j in range(S):
                            data[f'{k}_norm_eupnea_s{j+1}'] = y2_ds_by_key_norm_eupnea[k][:, j]
                    means_e, sems_e = self._nanmean_sem(y2_ds_by_key_norm_eupnea[k], axis=1)
                    data[f'{k}_norm_eupnea_mean'] = means_e
                    data[f'{k}_norm_eupnea_sem'] = sems_e

                # Create DataFrame and write to CSV
                df = pd.DataFrame(data, columns=header)
                df.to_csv(csv_time_path, index=False, float_format='%.9g', na_rep='')

            finally:
                self.window.unsetCursor()

            t_elapsed = time.time() - t_start
            print(f"[CSV] ✓ Time-series data written in {t_elapsed:.2f}s")

            # Save enhanced NPZ version 2 with timeseries data (for fast consolidation)
            _npz_timeseries_data['npz_version'] = 2
            _npz_timeseries_data['timeseries_t'] = t_ds_csv
            _npz_timeseries_data['timeseries_keys'] = list(keys_for_csv)
            # Save raw, norm, and eupnea-norm metric matrices
            for k in keys_for_csv:
                _npz_timeseries_data[f'ts_raw_{k}'] = y2_ds_by_key[k]
                _npz_timeseries_data[f'ts_norm_{k}'] = y2_ds_by_key_norm[k]
                _npz_timeseries_data[f'ts_eupnea_{k}'] = y2_ds_by_key_norm_eupnea[k]

            np.savez_compressed(npz_path, **_npz_timeseries_data)

            # Count total sniffing regions across all sweeps
            total_sniff_regions = sum(len(st.sniff_regions_by_sweep.get(s, [])) for s in kept_sweeps)
            print(f"[NPZ] ✓ Enhanced bundle saved (v2) with timeseries data")
            print(f"      - {total_sniff_regions} sniffing region(s) saved across {S} sweep(s)")

            if progress_dialog:
                progress_dialog.setLabelText("Writing breath-by-breath CSV...")
                progress_dialog.setValue(60)
                QApplication.processEvents()

            # -------------------- (3) Per-breath CSV (WIDE; with breath classifications) --------------------
            breaths_path = base.with_name(base.name + "_breaths.csv")

            BREATH_COLS = [
                "sweep", "breath", "t", "region", "is_sigh", "is_sniffing", "is_eupnea", "is_apnea",
                "if", "amp_insp", "amp_exp", "area_insp", "area_exp",
                "ti", "te", "vent_proxy",
            ]
            def _headers_for_block(suffix: str | None) -> list[str]:
                if not suffix: return BREATH_COLS[:]
                return [f"{c}_{suffix}" for c in BREATH_COLS]
    
            def _headers_for_block_norm(suffix: str | None) -> list[str]:
                base_cols = _headers_for_block(suffix)
                return [h + "_norm" for h in base_cols]

            def _headers_for_block_norm_eupnea(suffix: str | None) -> list[str]:
                base_cols = _headers_for_block(suffix)
                return [h + "_norm_eupnea" for h in base_cols]

            rows_all, rows_bl, rows_st, rows_po = [], [], [], []
            rows_all_N, rows_bl_N, rows_st_N, rows_po_N = [], [], [], []
            rows_all_E, rows_bl_E, rows_st_E, rows_po_E = [], [], [], []

            need_keys = ["if", "amp_insp", "amp_exp", "area_insp", "area_exp", "ti", "te", "vent_proxy"]

            # Compute EUPNEA-BASED baselines (pooled across all sweeps)
            # Pre-compute eupnea masks once per sweep to avoid redundant calculations
            eupnea_b_by_k = {}
            eupnea_masks_by_sweep = {}

            # Get thresholds from UI (for breath classification)
            eupnea_thresh = self.window.eupnea_freq_threshold  # Hz
            apnea_thresh = self.window._parse_float(self.window.ApneaThresh) or 0.5    # seconds

            t_start = time.time()
            print(f"[CSV] Pre-computing eupnea masks for {len(kept_sweeps)} sweeps...")
            for s in kept_sweeps:
                y_proc = self.window._get_processed_for(st.analyze_chan, s)
                pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                br = st.breath_by_sweep.get(s, None)
                if br is None and pks.size:
                    br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
                    st.breath_by_sweep[s] = br
                if br is None:
                    continue

                on = np.asarray(br.get("onsets", []), dtype=int)
                off = np.asarray(br.get("offsets", []), dtype=int)
                expmins = np.asarray(br.get("expmins", []), dtype=int)
                expoffs = np.asarray(br.get("expoffs", []), dtype=int)

                if on.size >= 2:
                    # Get sniff regions for this sweep
                    sniff_regions = st.sniff_regions_by_sweep.get(s, [])

                    # Compute eupnea mask (excluding sniffing regions)
                    eupnea_mask = metrics.detect_eupnic_regions(
                        st.t, y_proc, st.sr_hz, pks, on, off, expmins, expoffs,
                        freq_threshold_hz=eupnea_thresh,
                        min_duration_sec=self.window.eupnea_min_duration,
                        sniff_regions=sniff_regions
                    )
                    eupnea_masks_by_sweep[s] = eupnea_mask

            t_elapsed = time.time() - t_start
            print(f"[CSV] ✓ Eupnea masks computed in {t_elapsed:.2f}s")

            # Now compute baselines by collecting breath values from eupneic periods
            # IMPORTANT: Cache traces to reuse in main export loop AND PDF generation
            t_start = time.time()
            print(f"[CSV] Computing eupnea baselines and caching traces for {len(need_keys)} metrics...")
            eupnea_baseline_breaths = {k: [] for k in need_keys}

            # Export metric cache: Stores computed metric traces (sweep -> {metric: trace})
            # This cache is reused in PDF generation to avoid recomputing expensive metrics.
            # Note: This is separate from st.proc_cache which stores filtered raw traces.
            if not hasattr(self.window, '_export_metric_cache'):
                self.window._export_metric_cache = {}
            cached_traces_by_sweep = {}  # Cache traces to avoid recomputing

            for s in kept_sweeps:
                # Keep UI responsive during long computation
                if progress_dialog:
                    QApplication.processEvents()

                y_proc = self.window._get_processed_for(st.analyze_chan, s)
                pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                br = st.breath_by_sweep.get(s, None)
                if br is None or pks.size < 2:
                    continue

                on = np.asarray(br.get("onsets", []), dtype=int)
                if on.size < 2:
                    continue

                eupnea_mask = eupnea_masks_by_sweep.get(s, None)
                if eupnea_mask is None:
                    continue

                mids = (on[:-1] + on[1:]) // 2
                t_rel_all = (st.t[mids] - (global_s0 if have_global_stim else 0.0)).astype(float)

                # Compute traces once for this sweep and CACHE them (both locally and globally)
                traces_for_sweep = {}
                for k in need_keys:
                    if k in metrics.METRICS:
                        traces_for_sweep[k] = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, br)

                # Store in cache for reuse in main loop
                cached_traces_by_sweep[s] = traces_for_sweep
                # Also store globally for PDF reuse
                self.window._export_metric_cache[s] = traces_for_sweep

                # Collect baseline eupneic breath values
                for i, mid_idx in enumerate(mids):
                    if t_rel_all[i] >= 0:
                        continue  # Only baseline (t < 0)

                    # Check if this breath is eupneic
                    if mid_idx < len(eupnea_mask) and eupnea_mask[mid_idx] > 0:
                        for k in need_keys:
                            trace = traces_for_sweep.get(k, None)
                            if trace is not None and len(trace) == len(st.t):
                                val = trace[mid_idx]
                                if np.isfinite(val):
                                    eupnea_baseline_breaths[k].append(val)

            # Compute baseline means with fallback if insufficient baseline data
            for k in need_keys:
                if len(eupnea_baseline_breaths[k]) >= 10:
                    eupnea_b_by_k[k] = float(np.mean(eupnea_baseline_breaths[k]))
                elif len(eupnea_baseline_breaths[k]) > 0:
                    eupnea_b_by_k[k] = float(np.mean(eupnea_baseline_breaths[k]))
                else:
                    # Fallback: include post-stim eupneic breaths using CACHED traces
                    fallback_vals = []
                    for s in kept_sweeps:
                        traces_cached = cached_traces_by_sweep.get(s, None)
                        if traces_cached is None:
                            continue

                        br = st.breath_by_sweep.get(s, None)
                        if br is None:
                            continue

                        on = np.asarray(br.get("onsets", []), dtype=int)
                        if on.size < 2:
                            continue

                        eupnea_mask = eupnea_masks_by_sweep.get(s, None)
                        if eupnea_mask is None:
                            continue

                        mids = (on[:-1] + on[1:]) // 2
                        t_rel_all = (st.t[mids] - (global_s0 if have_global_stim else 0.0)).astype(float)

                        trace = traces_cached.get(k, None)
                        if trace is None or len(trace) != len(st.t):
                            continue

                        for i, mid_idx in enumerate(mids):
                            if 0 <= t_rel_all[i] <= NORM_BASELINE_WINDOW_S:
                                if mid_idx < len(eupnea_mask) and eupnea_mask[mid_idx] > 0:
                                    val = trace[mid_idx]
                                    if np.isfinite(val):
                                        fallback_vals.append(val)

                    eupnea_b_by_k[k] = float(np.mean(fallback_vals)) if len(fallback_vals) > 0 else np.nan

            t_elapsed = time.time() - t_start
            print(f"[CSV] ✓ Baselines and traces computed in {t_elapsed:.2f}s")

            t_start = time.time()
            print(f"[CSV] Writing breath-by-breath data...")
            for s in kept_sweeps:
                # Use cached traces if available, otherwise compute
                traces = cached_traces_by_sweep.get(s, None)
                if traces is None:
                    # Compute if not cached (shouldn't happen, but safety fallback)
                    y_proc = self.window._get_processed_for(st.analyze_chan, s)
                    pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                    br = st.breath_by_sweep.get(s, None)
                    if br is None and pks.size:
                        br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
                        st.breath_by_sweep[s] = br
                    if br is None:
                        br = {"onsets": np.array([], dtype=int)}

                    traces = {}
                    for k in need_keys:
                        if k in metrics.METRICS:
                            traces[k] = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, br)
                        else:
                            traces[k] = None
                else:
                    # Use cached data
                    br = st.breath_by_sweep.get(s, None)
                    if br is None:
                        br = {"onsets": np.array([], dtype=int)}

                on = np.asarray(br.get("onsets", []), dtype=int)
                if on.size < 2:
                    continue

                mids = (on[:-1] + on[1:]) // 2
    
                # Per-sweep breath-based baselines (use breath midpoints)
                t_rel_all = (st.t[mids] - (global_s0 if have_global_stim else 0.0)).astype(float)
                mask_pre_b  = (t_rel_all >= -NORM_BASELINE_WINDOW_S) & (t_rel_all < 0.0)
                mask_post_b = (t_rel_all >=  0.0) & (t_rel_all <= NORM_BASELINE_WINDOW_S)
    
                b_by_k = {}
                for k in need_keys:
                    arr = traces.get(k, None)
                    if arr is None or len(arr) != N:
                        b_by_k[k] = np.nan
                        continue
                    vals = arr[mids[mask_pre_b]]
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        vals = arr[mids[mask_post_b]]
                        vals = vals[np.isfinite(vals)]
                    b_by_k[k] = float(np.mean(vals)) if vals.size else np.nan
    
                # Breath classification flags: sigh, sniffing, eupnea, apnea
                # - is_sigh: Any sigh peak within breath interval [on[j], on[j+1])
                # - is_sniffing: Breath midpoint falls within marked sniffing region
                # - is_eupnea: Breath midpoint is part of eupneic (normal) breathing pattern
                # - is_apnea: Breath is preceded by long gap (recovery breath after apnea)

                # Sigh: Any sigh peak within breath interval [on[j], on[j+1])
                sigh_idx = np.asarray(st.sigh_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                sigh_idx = sigh_idx[(sigh_idx >= 0) & (sigh_idx < len(y_proc))]
                is_sigh_per_breath = np.zeros(on.size - 1, dtype=int)
                if sigh_idx.size:
                    for j in range(on.size - 1):
                        a = int(on[j]); b = int(on[j+1])
                        if np.any((sigh_idx >= a) & (sigh_idx < b)):
                            is_sigh_per_breath[j] = 1

                # Sniffing: Breath midpoint falls within any sniffing region
                sniff_regions = st.sniff_regions_by_sweep.get(s, [])
                is_sniffing_per_breath = np.zeros(on.size - 1, dtype=int)
                if sniff_regions:
                    for j, mid_idx in enumerate(mids):
                        t_mid = st.t[int(mid_idx)]
                        for sn_start, sn_end in sniff_regions:
                            if sn_start <= t_mid <= sn_end:
                                is_sniffing_per_breath[j] = 1
                                break

                # Eupnea: Breath midpoint marked as eupneic
                eupnea_mask = eupnea_masks_by_sweep.get(s, None)
                is_eupnea_per_breath = np.zeros(on.size - 1, dtype=int)
                if eupnea_mask is not None:
                    for j, mid_idx in enumerate(mids):
                        if int(mid_idx) < len(eupnea_mask) and eupnea_mask[int(mid_idx)] > 0:
                            is_eupnea_per_breath[j] = 1

                # Apnea: Breath preceded by long inter-breath interval (recovery breath after apnea)
                is_apnea_per_breath = np.zeros(on.size - 1, dtype=int)
                for j in range(len(mids)):
                    if j > 0:  # Need a previous onset to check inter-breath interval
                        ibi = st.t[int(on[j])] - st.t[int(on[j-1])]  # Time since last breath
                        if ibi > apnea_thresh:
                            # This breath (starting at on[j]) comes after a long gap
                            is_apnea_per_breath[j] = 1

                for i, idx in enumerate(mids, start=1):
                    t_rel = float(st.t[int(idx)] - (global_s0 if have_global_stim else 0.0))
                    breath_idx = i - 1  # 0-indexed for accessing arrays
                    sigh_flag = str(int(is_sigh_per_breath[breath_idx]))
                    sniff_flag = str(int(is_sniffing_per_breath[breath_idx]))
                    eupnea_flag = str(int(is_eupnea_per_breath[breath_idx]))
                    apnea_flag = str(int(is_apnea_per_breath[breath_idx]))
    
                    # ----- RAW: ALL
                    row_all = [str(s + 1), str(i), f"{t_rel:.9g}", "all", sigh_flag, sniff_flag, eupnea_flag, apnea_flag]
                    for k in need_keys:
                        v = np.nan
                        arr = traces.get(k, None)
                        if arr is not None and len(arr) == N:
                            v = arr[int(idx)]
                        row_all.append(f"{v:.9g}" if np.isfinite(v) else "")
                    rows_all.append(row_all)

                    # ----- NORM: ALL (time-based, per-sweep)
                    row_allN = [str(s + 1), str(i), f"{t_rel:.9g}", "all", sigh_flag, sniff_flag, eupnea_flag, apnea_flag]
                    for k in need_keys:
                        v = np.nan
                        arr = traces.get(k, None)
                        if arr is not None and len(arr) == N:
                            v = arr[int(idx)]
                        b = b_by_k.get(k, np.nan)
                        vn = (v / b) if (np.isfinite(v) and np.isfinite(b) and abs(b) > EPS_BASE) else np.nan
                        row_allN.append(f"{vn:.9g}" if np.isfinite(vn) else "")
                    rows_all_N.append(row_allN)

                    # ----- NORM_EUPNEA: ALL (eupnea-based, pooled)
                    row_allE = [str(s + 1), str(i), f"{t_rel:.9g}", "all", sigh_flag, sniff_flag, eupnea_flag, apnea_flag]
                    for k in need_keys:
                        v = np.nan
                        arr = traces.get(k, None)
                        if arr is not None and len(arr) == N:
                            v = arr[int(idx)]
                        eb = eupnea_b_by_k.get(k, np.nan)
                        ve = (v / eb) if (np.isfinite(v) and np.isfinite(eb) and abs(eb) > EPS_BASE) else np.nan
                        row_allE.append(f"{ve:.9g}" if np.isfinite(ve) else "")
                    rows_all_E.append(row_allE)

                    if have_global_stim:
                        region = "Baseline" if t_rel < 0 else ("Stim" if t_rel <= global_dur else "Post")

                        # RAW regional row
                        row_reg = [str(s + 1), str(i), f"{t_rel:.9g}", region, sigh_flag, sniff_flag, eupnea_flag, apnea_flag]
                        for k in need_keys:
                            v = np.nan
                            arr = traces.get(k, None)
                            if arr is not None and len(arr) == N:
                                v = arr[int(idx)]
                            row_reg.append(f"{v:.9g}" if np.isfinite(v) else "")
                        if region == "Baseline": rows_bl.append(row_reg)
                        elif region == "Stim":  rows_st.append(row_reg)
                        else:                   rows_po.append(row_reg)

                        # NORM regional row (time-based, per-sweep)
                        row_regN = [str(s + 1), str(i), f"{t_rel:.9g}", region, sigh_flag, sniff_flag, eupnea_flag, apnea_flag]
                        for k in need_keys:
                            v = np.nan
                            arr = traces.get(k, None)
                            if arr is not None and len(arr) == N:
                                v = arr[int(idx)]
                            b = b_by_k.get(k, np.nan)
                            vn = (v / b) if (np.isfinite(v) and np.isfinite(b) and abs(b) > EPS_BASE) else np.nan
                            row_regN.append(f"{vn:.9g}" if np.isfinite(vn) else "")
                        if region == "Baseline": rows_bl_N.append(row_regN)
                        elif region == "Stim":  rows_st_N.append(row_regN)
                        else:                   rows_po_N.append(row_regN)

                        # NORM_EUPNEA regional row (eupnea-based, pooled)
                        row_regE = [str(s + 1), str(i), f"{t_rel:.9g}", region, sigh_flag, sniff_flag, eupnea_flag, apnea_flag]
                        for k in need_keys:
                            v = np.nan
                            arr = traces.get(k, None)
                            if arr is not None and len(arr) == N:
                                v = arr[int(idx)]
                            eb = eupnea_b_by_k.get(k, np.nan)
                            ve = (v / eb) if (np.isfinite(v) and np.isfinite(eb) and abs(eb) > EPS_BASE) else np.nan
                            row_regE.append(f"{ve:.9g}" if np.isfinite(ve) else "")
                        if region == "Baseline": rows_bl_E.append(row_regE)
                        elif region == "Stim":  rows_st_E.append(row_regE)
                        else:                   rows_po_E.append(row_regE)
    
            def _pad_row(row, want_len):
                if row is None: return [""] * want_len
                if len(row) < want_len: return row + [""] * (want_len - len(row))
                return row
    
            headers_all = _headers_for_block(None)
            headers_bl  = _headers_for_block("baseline")
            headers_st  = _headers_for_block("stim")
            headers_po  = _headers_for_block("post")

            headers_allN = _headers_for_block_norm(None)
            headers_blN  = _headers_for_block_norm("baseline")
            headers_stN  = _headers_for_block_norm("stim")
            headers_poN  = _headers_for_block_norm("post")

            headers_allE = _headers_for_block_norm_eupnea(None)
            headers_blE  = _headers_for_block_norm_eupnea("baseline")
            headers_stE  = _headers_for_block_norm_eupnea("stim")
            headers_poE  = _headers_for_block_norm_eupnea("post")

            have_stim_blocks = have_global_stim and (len(rows_bl) + len(rows_st) + len(rows_po) > 0)

            # Build DataFrames from row lists (1.5-2× faster than row-by-row)
            import pandas as pd

            if not have_stim_blocks:
                # RAW + NORM + NORM_EUPNEA (ALL only)
                df_all = pd.DataFrame(rows_all, columns=headers_all) if rows_all else pd.DataFrame(columns=headers_all)
                df_all_N = pd.DataFrame(rows_all_N, columns=headers_allN) if rows_all_N else pd.DataFrame(columns=headers_allN)
                df_all_E = pd.DataFrame(rows_all_E, columns=headers_allE) if rows_all_E else pd.DataFrame(columns=headers_allE)

                # Add separator columns and concatenate
                sep1 = pd.DataFrame({'': [''] * len(df_all)}) if len(df_all) > 0 else pd.DataFrame({'': []})
                sep2 = pd.DataFrame({'': [''] * len(df_all)}) if len(df_all) > 0 else pd.DataFrame({'': []})

                df_combined = pd.concat([df_all, sep1, df_all_N, sep2, df_all_E], axis=1)
            else:
                # RAW blocks, then NORM blocks, then NORM_EUPNEA blocks
                df_all = pd.DataFrame(rows_all, columns=headers_all) if rows_all else pd.DataFrame(columns=headers_all)
                df_bl = pd.DataFrame(rows_bl, columns=headers_bl) if rows_bl else pd.DataFrame(columns=headers_bl)
                df_st = pd.DataFrame(rows_st, columns=headers_st) if rows_st else pd.DataFrame(columns=headers_st)
                df_po = pd.DataFrame(rows_po, columns=headers_po) if rows_po else pd.DataFrame(columns=headers_po)

                df_all_N = pd.DataFrame(rows_all_N, columns=headers_allN) if rows_all_N else pd.DataFrame(columns=headers_allN)
                df_bl_N = pd.DataFrame(rows_bl_N, columns=headers_blN) if rows_bl_N else pd.DataFrame(columns=headers_blN)
                df_st_N = pd.DataFrame(rows_st_N, columns=headers_stN) if rows_st_N else pd.DataFrame(columns=headers_stN)
                df_po_N = pd.DataFrame(rows_po_N, columns=headers_poN) if rows_po_N else pd.DataFrame(columns=headers_poN)

                df_all_E = pd.DataFrame(rows_all_E, columns=headers_allE) if rows_all_E else pd.DataFrame(columns=headers_allE)
                df_bl_E = pd.DataFrame(rows_bl_E, columns=headers_blE) if rows_bl_E else pd.DataFrame(columns=headers_blE)
                df_st_E = pd.DataFrame(rows_st_E, columns=headers_stE) if rows_st_E else pd.DataFrame(columns=headers_stE)
                df_po_E = pd.DataFrame(rows_po_E, columns=headers_poE) if rows_po_E else pd.DataFrame(columns=headers_poE)

                # Create separator columns (use max length for consistency)
                max_len = max(
                    len(df_all), len(df_bl), len(df_st), len(df_po),
                    len(df_all_N), len(df_bl_N), len(df_st_N), len(df_po_N),
                    len(df_all_E), len(df_bl_E), len(df_st_E), len(df_po_E)
                ) if any([len(df_all), len(df_bl), len(df_st), len(df_po)]) else 0

                sep = pd.DataFrame({'': [''] * max_len}) if max_len > 0 else pd.DataFrame({'': []})

                # Concatenate with separators
                df_combined = pd.concat([
                    df_all, sep.copy(), df_bl, sep.copy(), df_st, sep.copy(), df_po, sep.copy(),
                    df_all_N, sep.copy(), df_bl_N, sep.copy(), df_st_N, sep.copy(), df_po_N, sep.copy(),
                    df_all_E, sep.copy(), df_bl_E, sep.copy(), df_st_E, sep.copy(), df_po_E
                ], axis=1)

            # Write to CSV
            df_combined.to_csv(breaths_path, index=False, na_rep='')

            t_elapsed = time.time() - t_start
            print(f"[CSV] ✓ Breath data written in {t_elapsed:.2f}s")

            if progress_dialog:
                progress_dialog.setLabelText("Writing events CSV...")
                progress_dialog.setValue(70)
                QApplication.processEvents()

            # -------------------- (4) Events CSV (stimulus, apnea, eupnea intervals) --------------------
            t_start = time.time()
            events_path = base.with_name(base.name + "_events.csv")

            events_rows = []

            # Get thresholds from UI
            eupnea_thresh = self.window.eupnea_freq_threshold  # Hz
            apnea_thresh = self.window._parse_float(self.window.ApneaThresh) or 0.5    # seconds

            for s in kept_sweeps:
                y_proc = self.window._get_processed_for(st.analyze_chan, s)
                pks    = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                br     = st.breath_by_sweep.get(s, None)
                if br is None and pks.size:
                    br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
                if br is None:
                    br = {"onsets": np.array([], dtype=int)}

                on = np.asarray(br.get("onsets", []), dtype=int)
                off = np.asarray(br.get("offsets", []), dtype=int)
                expmins = np.asarray(br.get("expmins", []), dtype=int)
                expoffs = np.asarray(br.get("expoffs", []), dtype=int)

                # Stimulus events
                stim_on_idx = st.stim_onsets_by_sweep.get(s, None)
                stim_off_idx = st.stim_offsets_by_sweep.get(s, None)

                if stim_on_idx is not None and len(stim_on_idx) > 0:
                    for i, on_idx in enumerate(stim_on_idx):
                        start_time = float(st.t[int(on_idx)])
                        if stim_off_idx is not None and i < len(stim_off_idx):
                            end_time = float(st.t[int(stim_off_idx[i])])
                        else:
                            end_time = np.nan

                        # Convert to relative time if global stim available
                        if have_global_stim:
                            start_time -= global_s0
                            if np.isfinite(end_time):
                                end_time -= global_s0

                        duration = end_time - start_time if np.isfinite(end_time) else np.nan

                        events_rows.append([
                            str(s + 1),
                            "stimulus",
                            f"{start_time:.9g}",
                            f"{end_time:.9g}" if np.isfinite(end_time) else "",
                            f"{duration:.9g}" if np.isfinite(duration) else ""
                        ])

                # Apnea and eupnea events (only if we have breath data)
                if on.size >= 2:
                    # Detect apnea regions
                    apnea_mask = metrics.detect_apneas(
                        st.t, y_proc, st.sr_hz, pks, on, off, expmins, expoffs,
                        min_apnea_duration_sec=apnea_thresh
                    )

                    # Detect eupnea regions
                    eupnea_mask = metrics.detect_eupnic_regions(
                        st.t, y_proc, st.sr_hz, pks, on, off, expmins, expoffs,
                        freq_threshold_hz=eupnea_thresh
                    )

                    # Convert masks to interval lists
                    def mask_to_intervals(mask):
                        """Convert boolean mask to list of (start_idx, end_idx) intervals."""
                        intervals = []
                        in_region = False
                        start_idx = 0

                        for i in range(len(mask)):
                            if mask[i] and not in_region:
                                # Start of new region
                                start_idx = i
                                in_region = True
                            elif not mask[i] and in_region:
                                # End of region
                                intervals.append((start_idx, i - 1))
                                in_region = False

                        # Handle case where region extends to end
                        if in_region:
                            intervals.append((start_idx, len(mask) - 1))

                        return intervals

                    # Add apnea intervals
                    apnea_intervals = mask_to_intervals(apnea_mask > 0)
                    for start_idx, end_idx in apnea_intervals:
                        start_time = float(st.t[int(start_idx)])
                        end_time = float(st.t[int(end_idx)])

                        # Convert to relative time if global stim available
                        if have_global_stim:
                            start_time -= global_s0
                            end_time -= global_s0

                        duration = end_time - start_time

                        events_rows.append([
                            str(s + 1),
                            "apnea",
                            f"{start_time:.9g}",
                            f"{end_time:.9g}",
                            f"{duration:.9g}"
                        ])

                    # Add eupnea intervals
                    eupnea_intervals = mask_to_intervals(eupnea_mask > 0)
                    for start_idx, end_idx in eupnea_intervals:
                        start_time = float(st.t[int(start_idx)])
                        end_time = float(st.t[int(end_idx)])

                        # Convert to relative time if global stim available
                        if have_global_stim:
                            start_time -= global_s0
                            end_time -= global_s0

                        duration = end_time - start_time

                        events_rows.append([
                            str(s + 1),
                            "eupnea",
                            f"{start_time:.9g}",
                            f"{end_time:.9g}",
                            f"{duration:.9g}"
                        ])

                # Add sniffing bout intervals
                sniff_regions = st.sniff_regions_by_sweep.get(s, [])
                for start_time, end_time in sniff_regions:
                    # start_time and end_time are already in seconds

                    # Convert to relative time if global stim available
                    if have_global_stim:
                        start_time_rel = start_time - global_s0
                        end_time_rel = end_time - global_s0
                    else:
                        start_time_rel = start_time
                        end_time_rel = end_time

                    duration = end_time - start_time

                    events_rows.append([
                        str(s + 1),
                        "sniffing",
                        f"{start_time_rel:.9g}",
                        f"{end_time_rel:.9g}",
                        f"{duration:.9g}"
                    ])

            # Write events CSV using pandas
            import pandas as pd
            df_events = pd.DataFrame(
                events_rows,
                columns=["sweep", "event_type", "start_time", "end_time", "duration"]
            ) if events_rows else pd.DataFrame(columns=["sweep", "event_type", "start_time", "end_time", "duration"])
            df_events.to_csv(events_path, index=False, na_rep='')

            # Count event types
            if events_rows:
                event_counts = {}
                for row in events_rows:
                    event_type = row[1]
                    event_counts[event_type] = event_counts.get(event_type, 0) + 1
                event_summary = ", ".join([f"{count} {etype}" for etype, count in sorted(event_counts.items())])
            else:
                event_summary = "no events"

            t_elapsed = time.time() - t_start
            print(f"[CSV] ✓ Events data written in {t_elapsed:.2f}s ({event_summary})")

        # -------------------- (4) Summary PDF or Preview --------------------
        t_start = time.time()
        if progress_dialog:
            progress_dialog.setLabelText("Generating summary figures..." if preview_only else "Generating PDF...")
            progress_dialog.setValue(80)
            QApplication.processEvents()

        keys_for_timeplots = [k for k in all_keys if k not in self._EXCLUDE_FOR_CSV]
        label_by_key = {key: label for (label, key) in metrics.METRIC_SPECS if key in keys_for_timeplots}

        if preview_only:
            # Show interactive preview dialog instead of saving
            try:
                self._show_summary_preview_dialog(
                    t_ds_csv=t_ds_csv,
                    y2_ds_by_key=y2_ds_by_key,
                    keys_for_csv=keys_for_timeplots,
                    label_by_key=label_by_key,
                    stim_zero=(global_s0 if have_global_stim else None),
                    stim_dur=(global_dur if have_global_stim else None),
                )
            except Exception as e:
                QMessageBox.critical(self.window, "View Summary", f"Error generating preview:\n{e}")
                import traceback
                traceback.print_exc()
        else:
            # Save PDF to disk
            pdf_path = base.with_name(base.name + "_summary.pdf")
            try:
                self._save_metrics_summary_pdf(
                    out_path=pdf_path,
                    t_ds_csv=t_ds_csv,
                    y2_ds_by_key=y2_ds_by_key,
                    keys_for_csv=keys_for_timeplots,
                    label_by_key=label_by_key,
                    stim_zero=(global_s0 if have_global_stim else None),
                    stim_dur=(global_dur if have_global_stim else None),
                )
            except Exception as e:
                print(f"[save][summary-pdf] skipped: {e}")

            t_elapsed = time.time() - t_start
            if preview_only:
                print(f"[PDF] ✓ Preview generated in {t_elapsed:.2f}s")
            else:
                print(f"[PDF] ✓ PDF saved in {t_elapsed:.2f}s")

            # -------------------- done --------------------
            if progress_dialog:
                progress_dialog.setValue(100)
                QApplication.processEvents()

            msg = f"Saved:\n- {npz_path.name}\n- {csv_time_path.name}\n- {breaths_path.name}\n- {events_path.name}\n- {pdf_path.name}"
            print("[save]", msg)
            try:
                self.window.statusbar.showMessage(msg, 6000)
            except Exception:
                pass

            # Show success dialog
            QMessageBox.information(
                self.window,
                "Save Successful",
                f"Files saved successfully to:\n{self.window._save_dir}\n\n{msg}"
            )



    def _mean_sem_1d(self, arr: np.ndarray):
        """Finite-only mean and SEM (ddof=1) for a 1D array. Returns (mean, sem).
        If no finite values -> (nan, nan). If only 1 finite value -> (mean, nan)."""
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

   


    def _save_metrics_summary_pdf(
        self,
        out_path,
        t_ds_csv: np.ndarray,
        y2_ds_by_key: dict,
        keys_for_csv: list[str],
        label_by_key: dict[str, str],
        stim_zero: float | None,
        stim_dur: float | None,
        return_figures: bool = False,
         ):
        """
        Build a three-page PDF (or return figures for preview).

        If return_figures=True: Returns (fig1, fig2, fig3) tuple instead of saving.
        If return_figures=False: Saves PDF to out_path.

        • Page 1: rows = metrics, cols = [all sweeps | mean±SEM | histograms] using RAW data
        • Page 2: same layout, using NORMALIZED data (time-based, per-sweep baseline)
        • Page 3: same layout, using NORMALIZED data (eupnea-based, pooled baseline)
        • NEW: overlay orange star markers at times where sighs occurred (first two columns)
            and at x = metric value of sigh breaths (histogram column).
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        st = self.window.state
        n_sweeps_total = next(iter(y2_ds_by_key.values())).shape[1] if y2_ds_by_key else 0
        M = len(t_ds_csv)
        have_stim = (stim_zero is not None and stim_dur is not None)

        # --- normalization knobs (match exporter) ---
        NORM_BASELINE_WINDOW_S = float(getattr(self, "_norm_window_s", 10.0))
        EPS_BASE = 1e-12

        # ---------- Helpers ----------
        def _per_sweep_baseline_time(A_ds: np.ndarray) -> np.ndarray:
            b = np.full((A_ds.shape[1],), np.nan, dtype=float)
            mask_pre  = (t_ds_csv >= -NORM_BASELINE_WINDOW_S) & (t_ds_csv < 0.0)
            mask_post = (t_ds_csv >=  0.0) & (t_ds_csv <= NORM_BASELINE_WINDOW_S)
            for s in range(A_ds.shape[1]):
                col = A_ds[:, s]
                vals = col[mask_pre]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    vals = col[mask_post]
                    vals = vals[np.isfinite(vals)]
                if vals.size:
                    b[s] = float(np.mean(vals))
            return b

        def _normalize_matrix(A_ds: np.ndarray, b: np.ndarray) -> np.ndarray:
            out = np.full_like(A_ds, np.nan)
            for s in range(A_ds.shape[1]):
                bs = b[s]
                if np.isfinite(bs) and abs(bs) > EPS_BASE:
                    out[:, s] = A_ds[:, s] / bs
            return out

        def _plot_sigh_time_stars(ax, sigh_times_rel):
            """Overlay stars at sigh times near the top of a time-series axis without changing limits."""
            if not sigh_times_rel:
                return
            ylim = ax.get_ylim()
            y_star = ylim[1] - 0.06 * (ylim[1] - ylim[0])
            ax.plot(
                sigh_times_rel,
                [y_star] * len(sigh_times_rel),
                linestyle="none",
                marker="*",
                markersize=9,
                color="#ff9f1a",
                markeredgecolor="#a35400",
                alpha=0.95,
                zorder=6,
            )
            ax.set_ylim(*ylim)

        def _plot_hist_stars(ax, star_x_vals):
            """Overlay stars along the top of histogram axes at given x positions (metric values on sigh breaths)."""
            if not star_x_vals:
                return
            ylim = ax.get_ylim()
            y_star = ylim[1] - 0.06 * (ylim[1] - ylim[0])
            ax.scatter(
                np.asarray(star_x_vals, dtype=float),
                np.full(len(star_x_vals), y_star, dtype=float),
                marker="*",
                s=70,
                linewidths=0.9,
                edgecolors="#a35400",
                facecolors="#ff9f1a",
                zorder=6,
                clip_on=False,
            )
            ax.set_ylim(*ylim)

        # Build histogram pools AND collect (a) sigh times (for time-series stars)
        # and (b) metric values at sigh breaths (for histogram stars).
        def _build_hist_vals_raw_and_norm():
            hist_raw = {k: {"all": [], "baseline": [], "stim": [], "post": []} for k in keys_for_csv}
            hist_nrm = {k: {"all": [], "baseline": [], "stim": [], "post": []} for k in keys_for_csv}
            sigh_vals_raw_by_key  = {k: [] for k in keys_for_csv}   # x-positions for stars on histogram (raw)
            sigh_vals_norm_by_key = {k: [] for k in keys_for_csv}   # x-positions for stars on histogram (norm)
            sigh_times_rel = []  # for time-series stars (seconds, relative to stim_zero if provided)

            kept = [s for s in range(next(iter(st.sweeps.values())).shape[1])
                    if s not in getattr(st, "omitted_sweeps", set())]
            t0 = float(stim_zero) if stim_zero is not None else 0.0

            for s in kept:
                y_proc = self.window._get_processed_for(st.analyze_chan, s)
                pks    = np.asarray(self.window.state.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                br     = self.window.state.breath_by_sweep.get(s, None)
                if br is None and pks.size:
                    try:
                        br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
                    except TypeError:
                        br = peakdet.compute_breath_events(y_proc, pks)
                    self.window.state.breath_by_sweep[s] = br
                if br is None:
                    br = {"onsets": np.array([], dtype=int)}

                on = np.asarray(br.get("onsets", []), dtype=int)
                if on.size < 2:
                    continue

                mids = (on[:-1] + on[1:]) // 2
                t_rel_all = (st.t[mids] - t0).astype(float)

                # Whether each breath interval [on[j], on[j+1]) contains a sigh peak
                sigh_idx = np.asarray(self.window.state.sigh_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                is_sigh_breath = np.zeros(len(mids), dtype=bool)
                if sigh_idx.size:
                    for j in range(len(mids)):
                        a = int(on[j]); b = int(on[j + 1])
                        if np.any((sigh_idx >= a) & (sigh_idx < b)):
                            is_sigh_breath[j] = True
                            # time for time-series star (use breath midpoint)
                            sigh_times_rel.append(float(st.t[int(mids[j])] - t0))

                # Build metric traces - use global cache if available
                traces = None
                if hasattr(self.window, '_export_metric_cache'):
                    traces = self.window._export_metric_cache.get(s, None)

                if traces is None:
                    # Compute if not cached
                    traces = {}
                    for k in keys_for_csv:
                        try:
                            traces[k] = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, br)
                        except TypeError:
                            traces[k] = None

                # Per-sweep baselines for normalization (breath-based)
                mask_pre_b  = (t_rel_all >= -NORM_BASELINE_WINDOW_S) & (t_rel_all < 0.0)
                mask_post_b = (t_rel_all >=  0.0) & (t_rel_all <= NORM_BASELINE_WINDOW_S)
                b_by_k = {}
                for k in keys_for_csv:
                    arr = traces.get(k, None)
                    if arr is None or len(arr) != len(st.t):
                        b_by_k[k] = np.nan
                        continue
                    vals = arr[mids[mask_pre_b]]
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        vals = arr[mids[mask_post_b]]
                        vals = vals[np.isfinite(vals)]
                    b_by_k[k] = float(np.mean(vals)) if vals.size else np.nan

                # Fill histogram pools & record sigh metric values
                for j, (idx_mid, t_rel) in enumerate(zip(mids, t_rel_all)):
                    for k in keys_for_csv:
                        arr = traces.get(k, None)
                        if arr is None or len(arr) != len(st.t):
                            continue
                        v = float(arr[int(idx_mid)])
                        if not np.isfinite(v):
                            continue

                        # RAW pools
                        hist_raw[k]["all"].append(v)
                        if have_stim:
                            if t_rel < 0:
                                hist_raw[k]["baseline"].append(v)
                            elif 0.0 <= t_rel <= stim_dur:
                                hist_raw[k]["stim"].append(v)
                            else:
                                hist_raw[k]["post"].append(v)

                        # NORM pools
                        b = b_by_k.get(k, np.nan)
                        vn = (v / b) if (np.isfinite(b) and abs(b) > EPS_BASE) else np.nan
                        if np.isfinite(vn):
                            hist_nrm[k]["all"].append(vn)
                            if have_stim:
                                if t_rel < 0:
                                    hist_nrm[k]["baseline"].append(vn)
                                elif 0.0 <= t_rel <= stim_dur:
                                    hist_nrm[k]["stim"].append(vn)
                                else:
                                    hist_nrm[k]["post"].append(vn)

                        # SIGH stars (hist column): collect x = metric values on sigh breaths
                        if is_sigh_breath[j]:
                            sigh_vals_raw_by_key[k].append(v)
                            if np.isfinite(vn):
                                sigh_vals_norm_by_key[k].append(vn)

            # Dedup very-close times for cleaner star rows in time plots
            if len(st.t) > 1:
                dt = float(np.median(np.diff(st.t)))
            else:
                dt = (1.0 / float(st.sr_hz)) if getattr(st, "sr_hz", None) else 0.01
            eps = max(1e-9, 2.0 * dt)
            sigh_times_rel = sorted(sigh_times_rel)
            sigh_times_clean = []
            for tt in sigh_times_rel:
                if not sigh_times_clean or abs(tt - sigh_times_clean[-1]) > eps:
                    sigh_times_clean.append(tt)

            return hist_raw, hist_nrm, sigh_vals_raw_by_key, sigh_vals_norm_by_key, sigh_times_clean

        def _rowwise_mean_sem(Y2D: np.ndarray):
                    import numpy as np
                    # Y2D shape: (M, S) = (time, sweeps)
                    if Y2D is None or Y2D.size == 0:
                        M = 0
                        return np.array([]), np.array([])
                    M = Y2D.shape[0]
                    n = np.sum(np.isfinite(Y2D), axis=1)          # (M,)
                    mean = np.full(M, np.nan, float)
                    sem  = np.full(M, np.nan, float)

                    msk_mean = n > 0
                    if np.any(msk_mean):
                        mean[msk_mean] = np.nanmean(Y2D[msk_mean, :], axis=1)

                    msk_sem = n >= 2
                    if np.any(msk_sem):
                        std = np.nanstd(Y2D[msk_sem, :], axis=1, ddof=1)
                        sem[msk_sem] = std / np.sqrt(n[msk_sem])
                    return mean, sem

        def _plot_grid(fig, axes, Y_by_key, hist_vals, sigh_hist_vals_by_key, sigh_times_rel, title_suffix):
            """Render one page (grid) given series & histogram data dicts + star overlays."""
            nrows = max(1, len(keys_for_csv))
            for r, k in enumerate(keys_for_csv):
                label = label_by_key.get(k, k)
                is_bottom_row = (r == nrows - 1)
                is_top_row = (r == 0)

                # --- col 1: all sweeps overlaid ---
                ax1 = axes[r, 0]
                Y = Y_by_key.get(k, None)
                if Y is not None and Y.shape[0] == M:
                    for s in range(Y.shape[1]):
                        ax1.plot(t_ds_csv, Y[:, s], lw=0.8, alpha=0.45)
                if have_stim:
                    ax1.axvspan(0.0, stim_dur, color="#2E5090", alpha=0.25)
                    ax1.axvline(0.0, color="#2E5090", lw=1.0, alpha=0.8)
                    ax1.axvline(stim_dur, color="#2E5090", lw=1.0, alpha=0.8, ls="--")
                _plot_sigh_time_stars(ax1, sigh_times_rel)
                # Use less padding on top row to make room for figure title
                title_pad = 3 if is_top_row else 8
                ax1.set_title(f"{label} — all sweeps{title_suffix}", fontsize=9, pad=title_pad)
                ax1.set_ylabel(label, fontsize=8)  # Add y-label with metric name
                ax1.set_xlabel("Time (s)", fontsize=8)

                # --- col 2: mean ± SEM ---
                ax2 = axes[r, 1]
                if Y is not None and Y.shape[0] == M:
                    mean, sem = _rowwise_mean_sem(Y)
                    ax2.plot(t_ds_csv, mean, lw=1.8)
                    ax2.fill_between(t_ds_csv, mean - sem, mean + sem, alpha=0.25, linewidth=0)

                if have_stim:
                    ax2.axvspan(0.0, stim_dur, color="#2E5090", alpha=0.25)
                    ax2.axvline(0.0, color="#2E5090", lw=1.0, alpha=0.8)
                    ax2.axvline(stim_dur, color="#2E5090", lw=1.0, alpha=0.8, ls="--")
                _plot_sigh_time_stars(ax2, sigh_times_rel)
                ax2.set_title(f"{label} — mean ± SEM{title_suffix}", fontsize=9, pad=title_pad)
                ax2.set_ylabel(label, fontsize=8)  # Add y-label with metric name
                ax2.set_xlabel("Time (s)", fontsize=8)

                # --- col 3: line histograms (density) + stars at sigh metric values ---
                ax3 = axes[r, 2]
                groups = []
                for nm in ("all", "baseline", "stim", "post"):
                    vals = np.asarray(hist_vals[k][nm], dtype=float)
                    if vals.size:
                        groups.append(vals)

                if len(groups):
                    combined = np.concatenate(groups)
                    edges = np.histogram_bin_edges(combined, bins="auto")
                    centers = 0.5 * (edges[:-1] + edges[1:])

                    def _plot_line(vals, lbl, style_kw):
                        vals = np.asarray(vals, dtype=float)
                        if vals.size == 0:
                            return
                        dens, _ = np.histogram(vals, bins=edges, density=True)
                        ax3.plot(centers, dens, **style_kw, label=lbl)

                    _plot_line(hist_vals[k]["all"], "All", dict(lw=1.8))
                    if have_stim:
                        _plot_line(hist_vals[k]["baseline"], "Baseline", dict(lw=1.6))
                        _plot_line(hist_vals[k]["stim"],     "Stim",     dict(lw=1.6, ls="--"))
                        _plot_line(hist_vals[k]["post"],     "Post",     dict(lw=1.6, ls=":"))

                ax3.set_title(f"{label} — distribution (density){title_suffix}", fontsize=9, pad=title_pad)
                ax3.set_ylabel("Density", fontsize=8)
                ax3.set_xlabel(label, fontsize=8)  # Add metric name as x-label
                # Only show legend on top row to save space
                if len(ax3.lines) and is_top_row:
                    ax3.legend(loc="best", fontsize=7)

                # NEW: stars for histogram at sigh metric values (use "all" sigh values)
                _plot_hist_stars(ax3, sigh_hist_vals_by_key.get(k, []))

                # Reduce tick label font size for all axes
                ax1.tick_params(labelsize=7)
                ax2.tick_params(labelsize=7)
                ax3.tick_params(labelsize=7)

            fig.tight_layout()

        # ---------- Prepare both RAW and NORMALIZED datasets ----------
        y2_ds_by_key_norm = {}
        for k in keys_for_csv:
            Y = y2_ds_by_key.get(k, None)
            if Y is None or not Y.size:
                y2_ds_by_key_norm[k] = None
                continue
            b = _per_sweep_baseline_time(Y)
            y2_ds_by_key_norm[k] = _normalize_matrix(Y, b)

        # Prepare EUPNEA-BASED normalized datasets
        y2_ds_by_key_norm_eupnea = {}
        eupnea_baselines_by_key = {}  # Store computed baselines for histogram building
        eupnea_thresh = self.window.eupnea_freq_threshold  # Hz

        # OPTIMIZATION: Compute eupnea masks once per sweep, reuse for all metrics
        eupnea_masks_by_sweep = {}
        kept = [s for s in range(next(iter(st.sweeps.values())).shape[1])
                if s not in getattr(st, "omitted_sweeps", set())]

        print(f"[PDF] Computing eupnea masks for {len(kept)} sweeps...")
        for s in kept:
            y_proc = self.window._get_processed_for(st.analyze_chan, s)
            pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
            br = st.breath_by_sweep.get(s, None)
            if br is None and pks.size:
                try:
                    br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
                except TypeError:
                    br = peakdet.compute_breath_events(y_proc, pks)

            if br is not None:
                on = np.asarray(br.get("onsets", []), dtype=int)
                off = np.asarray(br.get("offsets", []), dtype=int)
                expmins = np.asarray(br.get("expmins", []), dtype=int)
                expoffs = np.asarray(br.get("expoffs", []), dtype=int)

                if on.size >= 2:
                    eupnea_mask = metrics.detect_eupnic_regions(
                        st.t, y_proc, st.sr_hz, pks, on, off, expmins, expoffs,
                        freq_threshold_hz=eupnea_thresh,
                        min_duration_sec=self.window.eupnea_min_duration
                    )
                    eupnea_masks_by_sweep[s] = eupnea_mask

        print(f"[PDF] Computed {len(eupnea_masks_by_sweep)} eupnea masks")

        print(f"[PDF] Building histograms (raw and time-normalized)...")
        # Pools + sigh overlays
        (hist_vals_raw,
        hist_vals_norm,
        sigh_vals_raw_by_key,
        sigh_vals_norm_by_key,
        sigh_times_rel) = _build_hist_vals_raw_and_norm()
        print(f"[PDF] Built raw and time-normalized histograms")

        # Build EUPNEA-normalized histograms using a simple breath-based approach
        print(f"[PDF] Building eupnea-normalized histograms...")

        def _build_eupnea_normalized_hists():
            """
            Simple approach:
            1. Collect all breath metric values from eupneic baseline periods
            2. Compute mean baseline per metric
            3. Normalize all breath values by that baseline
            """
            # Step 1: Collect eupneic baseline breath values
            eupnea_baseline_breaths = {k: [] for k in keys_for_csv}
            all_breath_values_raw = {k: {"all": [], "baseline": [], "stim": [], "post": []} for k in keys_for_csv}
            sigh_vals_raw = {k: [] for k in keys_for_csv}

            kept = [s for s in range(next(iter(st.sweeps.values())).shape[1])
                    if s not in getattr(st, "omitted_sweeps", set())]
            t0 = float(stim_zero) if stim_zero is not None else 0.0

            for s in kept:
                y_proc = self.window._get_processed_for(st.analyze_chan, s)
                pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                br = st.breath_by_sweep.get(s, None)
                if br is None or pks.size < 2:
                    continue

                on = np.asarray(br.get("onsets", []), dtype=int)
                if on.size < 2:
                    continue

                # Get eupnea mask for this sweep
                eupnea_mask = eupnea_masks_by_sweep.get(s, None)
                if eupnea_mask is None:
                    continue

                mids = (on[:-1] + on[1:]) // 2
                t_rel_all = (st.t[mids] - t0).astype(float)

                # Check which breaths are sighs
                sigh_idx = np.asarray(st.sigh_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                is_sigh_breath = np.zeros(len(mids), dtype=bool)
                if sigh_idx.size:
                    for j in range(len(mids)):
                        a = int(on[j]); b = int(on[j + 1])
                        if np.any((sigh_idx >= a) & (sigh_idx < b)):
                            is_sigh_breath[j] = True

                # For each breath, check if midpoint is eupneic and collect metric values
                for i, mid_idx in enumerate(mids):
                    t_rel = t_rel_all[i]
                    is_eupneic = eupnea_mask[mid_idx] > 0 if mid_idx < len(eupnea_mask) else False
                    is_baseline = t_rel < 0

                    # Determine region
                    if have_stim:
                        if t_rel < 0:
                            region = "baseline"
                        elif t_rel <= stim_dur:
                            region = "stim"
                        else:
                            region = "post"
                    else:
                        region = "all"

                    # Extract metric values from raw histograms (already computed by _build_hist_vals_raw_and_norm)
                    # We'll use the same raw values that were already extracted
                    # But we need to recompute - let's just use the data from hist_vals_raw
                    # Actually, we can't easily map back. Let's sample from y2_ds_by_key at this time point

                    for k in keys_for_csv:
                        Y_raw = y2_ds_by_key.get(k, None)
                        if Y_raw is None or Y_raw.size == 0:
                            continue

                        # Find closest downsampled time point
                        ds_idx = np.argmin(np.abs(t_ds_csv - t_rel))
                        if ds_idx >= Y_raw.shape[0]:
                            continue

                        col_idx = kept.index(s)
                        val = Y_raw[ds_idx, col_idx]

                        if not np.isfinite(val):
                            continue

                        # Collect for baseline calculation if eupneic and baseline
                        if is_eupneic and is_baseline:
                            eupnea_baseline_breaths[k].append(val)

                        # Collect all breath values (will normalize later)
                        all_breath_values_raw[k]["all"].append(val)
                        if have_stim:
                            all_breath_values_raw[k][region].append(val)

                        # Collect sigh values
                        if is_sigh_breath[i]:
                            sigh_vals_raw[k].append(val)

            # Step 2: Compute eupneic baseline means
            eupnea_baselines = {}
            for k in keys_for_csv:
                if len(eupnea_baseline_breaths[k]) >= 10:
                    eupnea_baselines[k] = np.mean(eupnea_baseline_breaths[k])
                elif len(eupnea_baseline_breaths[k]) > 0:
                    # Use what we have if < 10
                    eupnea_baselines[k] = np.mean(eupnea_baseline_breaths[k])
                else:
                    eupnea_baselines[k] = np.nan

            # Step 3: Normalize all values
            hist_norm_eupnea = {k: {"all": [], "baseline": [], "stim": [], "post": []} for k in keys_for_csv}
            sigh_vals_norm_eupnea = {k: [] for k in keys_for_csv}

            for k in keys_for_csv:
                baseline = eupnea_baselines.get(k, np.nan)
                if not np.isfinite(baseline) or abs(baseline) < EPS_BASE:
                    continue

                for region in ["all", "baseline", "stim", "post"]:
                    hist_norm_eupnea[k][region] = [v / baseline for v in all_breath_values_raw[k][region]]

                sigh_vals_norm_eupnea[k] = [v / baseline for v in sigh_vals_raw[k]]

            # Also store baselines and normalized time series for plotting
            y2_norm_eupnea = {}
            for k in keys_for_csv:
                Y_raw = y2_ds_by_key.get(k, None)
                baseline = eupnea_baselines.get(k, np.nan)
                if Y_raw is not None and np.isfinite(baseline) and abs(baseline) > EPS_BASE:
                    y2_norm_eupnea[k] = Y_raw / baseline
                else:
                    y2_norm_eupnea[k] = None

            return hist_norm_eupnea, sigh_vals_norm_eupnea, eupnea_baselines, y2_norm_eupnea

        hist_vals_norm_eupnea, sigh_vals_norm_eupnea_by_key, eupnea_baselines_by_key, y2_ds_by_key_norm_eupnea = _build_eupnea_normalized_hists()
        print(f"[PDF] Built eupnea-normalized histograms")

        # ---------- Create three-page PDF ----------
        nrows = max(1, len(keys_for_csv))
        fig_w = 13
        fig_h = max(4.0, 2.6 * nrows)  # Reduced from 5.25 to 2.6 per row (half of 5.25)

        # Page 1 — RAW
        fig1, axes1 = plt.subplots(nrows, 3, figsize=(fig_w, fig_h), squeeze=False)
        _plot_grid(fig1, axes1, y2_ds_by_key, hist_vals_raw, sigh_vals_raw_by_key, sigh_times_rel, title_suffix="")
        fig1.suptitle("Summary — raw", fontsize=11)
        fig1.tight_layout(rect=[0, 0, 1, 0.99])  # Leave 1% at top for suptitle

        # Page 2 — NORMALIZED (time-based)
        fig2, axes2 = plt.subplots(nrows, 3, figsize=(fig_w, fig_h), squeeze=False)
        _plot_grid(fig2, axes2, y2_ds_by_key_norm, hist_vals_norm, sigh_vals_norm_by_key, sigh_times_rel, title_suffix=" (norm)")
        fig2.suptitle("Summary — normalized (time-based)", fontsize=11)
        fig2.tight_layout(rect=[0, 0, 1, 0.99])  # Leave 1% at top for suptitle

        # Page 3 — NORMALIZED (eupnea-based)
        fig3, axes3 = plt.subplots(nrows, 3, figsize=(fig_w, fig_h), squeeze=False)
        _plot_grid(fig3, axes3, y2_ds_by_key_norm_eupnea, hist_vals_norm_eupnea, sigh_vals_norm_eupnea_by_key, sigh_times_rel, title_suffix=" (norm eupnea)")
        fig3.suptitle("Summary — normalized (eupnea-based)", fontsize=11)
        fig3.tight_layout(rect=[0, 0, 1, 0.99])  # Leave 1% at top for suptitle

        # Either return figures or save to PDF
        if return_figures:
            return fig1, fig2, fig3
        else:
            with PdfPages(out_path) as pdf:
                pdf.savefig(fig1, dpi=150)
                pdf.savefig(fig2, dpi=150)
                pdf.savefig(fig3, dpi=150)
            plt.close(fig1)
            plt.close(fig2)
            plt.close(fig3)



    def _show_summary_preview_dialog(self, t_ds_csv, y2_ds_by_key, keys_for_csv, label_by_key, stim_zero, stim_dur):
        """Display interactive preview dialog with the three summary figures."""
        import matplotlib.pyplot as plt
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

        # Generate figures using the same method that saves PDFs, but get figures back instead
        fig1, fig2, fig3 = self._save_metrics_summary_pdf(
            out_path=None,  # Not used when return_figures=True
            t_ds_csv=t_ds_csv,
            y2_ds_by_key=y2_ds_by_key,
            keys_for_csv=keys_for_csv,
            label_by_key=label_by_key,
            stim_zero=stim_zero,
            stim_dur=stim_dur,
            return_figures=True,
        )

        # -------------------- Display in dialog --------------------
        dialog = QDialog(self.window)
        dialog.setWindowTitle("Summary Preview")
        dialog.resize(1300, 900)  # Slightly larger window

        main_layout = QVBoxLayout(dialog)

        # Page selector controls at top
        control_layout = QHBoxLayout()
        page_label = QLabel("Page 1 of 3 (Raw)")
        prev_btn = QPushButton("← Previous")
        next_btn = QPushButton("Next →")
        close_btn = QPushButton("Close")

        control_layout.addWidget(prev_btn)
        control_layout.addWidget(page_label)
        control_layout.addWidget(next_btn)
        control_layout.addStretch()
        control_layout.addWidget(close_btn)

        main_layout.addLayout(control_layout)

        # Create scroll area for the canvas with mouse wheel support
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Allow canvas to resize to fit width
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Enable mouse wheel scrolling by setting focus policy
        scroll_area.setFocusPolicy(Qt.FocusPolicy.WheelFocus)

        # Canvas for displaying matplotlib figures
        canvas1 = FigureCanvas(fig1)
        canvas2 = FigureCanvas(fig2)
        canvas3 = FigureCanvas(fig3)

        # Set size policies to allow width scaling but keep height fixed
        from PyQt6.QtWidgets import QSizePolicy
        from PyQt6.QtCore import QEvent
        for canvas in [canvas1, canvas2, canvas3]:
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            canvas.setMinimumWidth(800)  # Minimum width

        # Create a container widget to hold all three canvases (so they don't get deleted)
        from PyQt6.QtWidgets import QWidget, QStackedWidget
        canvas_stack = QStackedWidget()
        canvas_stack.addWidget(canvas1)  # index 0
        canvas_stack.addWidget(canvas2)  # index 1
        canvas_stack.addWidget(canvas3)  # index 2
        canvas_stack.setCurrentIndex(0)  # Start with page 1

        # Install event filter on canvases to forward wheel events to scroll area
        class WheelEventFilter:
            def __init__(self, scroll_area):
                self.scroll_area = scroll_area

            def eventFilter(self, obj, event):
                if event.type() == QEvent.Type.Wheel:
                    # Forward wheel event to scroll area's viewport
                    scrollbar = self.scroll_area.verticalScrollBar()
                    scrollbar.setValue(scrollbar.value() - event.angleDelta().y())
                    return True  # Event handled
                return False  # Let other events pass through

        wheel_filter = WheelEventFilter(scroll_area)
        from PyQt6.QtCore import QObject

        class EventFilterObject(QObject):
            def __init__(self, filter_func):
                super().__init__()
                self.filter_func = filter_func

            def eventFilter(self, obj, event):
                return self.filter_func(obj, event)

        filter_obj = EventFilterObject(wheel_filter.eventFilter)
        canvas1.installEventFilter(filter_obj)
        canvas2.installEventFilter(filter_obj)
        canvas3.installEventFilter(filter_obj)

        # Put the stacked widget in the scroll area
        scroll_area.setWidget(canvas_stack)
        main_layout.addWidget(scroll_area)

        # Page navigation
        current_page = [1]  # Use list to allow modification in nested function

        def update_page():
            if current_page[0] == 1:
                canvas_stack.setCurrentIndex(0)  # Show canvas1
                page_label.setText("Page 1 of 3 (Raw)")
                prev_btn.setEnabled(False)
                next_btn.setEnabled(True)
            elif current_page[0] == 2:
                canvas_stack.setCurrentIndex(1)  # Show canvas2
                page_label.setText("Page 2 of 3 (Normalized - Time-based)")
                prev_btn.setEnabled(True)
                next_btn.setEnabled(True)
            else:
                canvas_stack.setCurrentIndex(2)  # Show canvas3
                page_label.setText("Page 3 of 3 (Normalized - Eupnea-based)")
                prev_btn.setEnabled(True)
                next_btn.setEnabled(False)
            # Reset scroll position to top when changing pages
            scroll_area.verticalScrollBar().setValue(0)

        def go_prev():
            current_page[0] = max(1, current_page[0] - 1)
            update_page()

        def go_next():
            current_page[0] = min(3, current_page[0] + 1)
            update_page()

        prev_btn.clicked.connect(go_prev)
        next_btn.clicked.connect(go_next)
        close_btn.clicked.connect(dialog.close)

        update_page()
        dialog.exec()

        # Cleanup
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)



    def _sigh_sample_indices(self, s: int, pks: np.ndarray | None) -> set[int]:
        """
        Return a set of SAMPLE indices (into st.t / y) for sigh-marked peaks on sweep s,
        regardless of how they were originally stored.

        Accepts any of these storage patterns per sweep:
        • sample indices (ints 0..N-1)
        • indices INTO the peaks list (ints 0..len(pks)-1), which we map via pks[idx]
        • times in seconds (floats), which we map to nearest sample via searchsorted
        • numpy array / list / set in any of the above forms
        """
        st = self.window.state
        N = len(st.t)

        # Prefer the name you've been using; try some alternates if needed.
        candidates = None
        for name in ("sighs_by_sweep", "sigh_indices_by_sweep", "sigh_peaks_by_sweep", "sigh_mask_by_sweep"):
            if hasattr(st, name):
                candidates = getattr(st, name).get(s, None)
                if candidates is not None:
                    break

        if candidates is None:
            return set()

        arr = np.asarray(list(candidates))
        out: set[int] = set()

        # If integer-like
        if arr.dtype.kind in "iu":
            arr = arr.astype(int)
            if pks is not None and arr.size and arr.max(initial=-1) < len(pks):
                # Looks like indexes INTO peaks -> map to sample indexes
                for idx in arr:
                    if 0 <= idx < len(pks):
                        i = int(pks[idx])
                        if 0 <= i < N:
                            out.add(i)
            else:
                # Already sample indexes
                for i in arr:
                    if 0 <= i < N:
                        out.add(int(i))
            return out

        # If float-like -> assume times (seconds)
        if arr.dtype.kind in "f":
            t = st.t
            for val in arr:
                try:
                    i = int(np.clip(np.searchsorted(t, float(val)), 0, N - 1))
                    out.add(i)
                except Exception:
                    pass
            return out

        # Fallback: try to coerce each element
        for v in arr:
            try:
                i = int(v)
                if 0 <= i < N:
                    out.add(i)
            except Exception:
                try:
                    i = int(np.clip(np.searchsorted(st.t, float(v)), 0, N - 1))
                    out.add(i)
                except Exception:
                    pass
        return out



    ##################################################
    ##Curration                                     ##
    ##################################################
    #     if not base:
    #         return

