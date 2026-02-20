"""
Scan Manager - Handles file scanning and metadata loading for Project Builder.

Extracted from main.py to improve maintainability.
Contains:
- File scanning and discovery
- Background metadata loading with QThread
- Saved data scanning (NPZ files)
- Incomplete metadata reloading
"""

from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Any, Optional, Set
from datetime import datetime
import re

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QProgressDialog

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QMainWindow


class MetadataThread(QThread):
    """Background thread for loading file metadata using parallel processing."""

    # Batch updates: send list of results instead of individual items
    batch_progress = pyqtSignal(list, int, int)  # [(index, metadata), ...], total, completed_so_far
    finished = pyqtSignal(set)  # protocols

    def __init__(self, files: List[Path]):
        super().__init__()
        self.files = files
        self.should_stop = False

    def run(self):
        from core.fast_abf_reader import read_metadata_parallel

        protocols = set()
        batch = []
        batch_size = 25  # Update UI every 25 files
        completed_count = [0]

        def callback(index, total, metadata):
            if self.should_stop:
                return

            if metadata:
                protocols.add(metadata.get('protocol', 'Unknown'))

            batch.append((index, metadata))
            completed_count[0] += 1

            if len(batch) >= batch_size or completed_count[0] == total:
                self.batch_progress.emit(batch[:], total, completed_count[0])
                batch.clear()

        try:
            read_metadata_parallel(self.files, progress_callback=callback, max_workers=4)

            if batch:
                self.batch_progress.emit(batch[:], len(self.files), completed_count[0])
                batch.clear()

            self.finished.emit(protocols)
        except Exception as e:
            print(f"[scan-manager] Error during parallel loading: {e}")
            import traceback
            traceback.print_exc()
            self.finished.emit(protocols)


class ReloadMetadataThread(QThread):
    """Background thread for reloading incomplete metadata."""

    batch_progress = pyqtSignal(list, int, int)  # [(orig_index, row_index, metadata), ...], total, completed
    finished = pyqtSignal(set)

    def __init__(self, file_paths: List[Path], row_indices: List[int]):
        super().__init__()
        self.file_paths = file_paths
        self.row_indices = row_indices

    def run(self):
        from core.fast_abf_reader import read_file_metadata_fast

        protocols = set()
        results = []

        for i, (file_path, row_idx) in enumerate(zip(self.file_paths, self.row_indices)):
            try:
                metadata = read_file_metadata_fast(file_path)
                if metadata:
                    protocols.add(metadata.get('protocol', 'Unknown'))
                results.append((i, row_idx, metadata))
            except Exception as e:
                print(f"[scan-manager] Error reading {file_path}: {e}")
                results.append((i, row_idx, None))

            if len(results) % 10 == 0 or i == len(self.file_paths) - 1:
                self.batch_progress.emit(results[:], len(self.file_paths), len(results))
                results.clear()

        if results:
            self.batch_progress.emit(results, len(self.file_paths), len(self.file_paths))

        self.finished.emit(protocols)


class ScanManager:
    """Manages file scanning and metadata loading for Project Builder.

    Handles:
    - Directory scanning for data files (ABF, SMRX, EDF, photometry)
    - Background metadata loading with progress updates
    - Saved data (NPZ) scanning and matching
    - Incomplete metadata reloading
    """

    def __init__(self, main_window: 'QMainWindow'):
        """Initialize ScanManager.

        Args:
            main_window: Reference to MainWindow for widget access
        """
        self.mw = main_window
        self._metadata_thread = None
        self._reload_thread = None
        self._metadata_row_offset = 0

    def on_project_scan_files(self):
        """Scan directory for new files - additive mode, preserves existing data."""
        if not self.mw._project_directory:
            self.mw._show_warning("No Directory", "Please select a directory first using 'Browse Directory'.")
            return

        from core import project_builder
        from core.fast_abf_reader import extract_path_keywords

        # Prevent multiple concurrent scans
        if self._metadata_thread and self._metadata_thread.isRunning():
            self.mw._show_warning("Scan In Progress", "A scan is already running. Please wait for it to complete.")
            return

        progress = None
        try:
            # Disable scan button during operation
            self.mw.scanFilesButton.setEnabled(False)

            recursive = self.mw.recursiveCheckBox.isChecked()
            file_types = self.mw._project_builder.get_selected_file_types()

            # PHASE 1: Quick file discovery
            progress = QProgressDialog("Finding files...", "Cancel", 0, 0, self.mw)
            progress.setWindowTitle("Scanning Directory")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.show()
            QApplication.processEvents()

            files = project_builder.discover_files(
                self.mw._project_directory, recursive=recursive, file_types=file_types
            )
            abf_files = files['abf_files']
            smrx_files = files.get('smrx_files', [])
            edf_files = files.get('edf_files', [])
            photometry_files = files.get('photometry_files', [])
            notes_files = files.get('notes_files', [])

            all_data_files = abf_files + smrx_files + edf_files

            if progress.wasCanceled():
                progress.close()
                self.mw.scanFilesButton.setEnabled(True)
                self.mw._log_status_message("Scan cancelled", 2000)
                return

            progress.close()

            # Store notes files
            self.mw._discovered_notes_files = notes_files
            if notes_files:
                print(f"[scan-manager] Found {len(notes_files)} notes files")
            if photometry_files:
                print(f"[scan-manager] Found {len(photometry_files)} photometry files")

            # PHASE 2: Build set of existing file paths to avoid duplicates
            existing_paths = set()
            for task in self.mw._master_file_list:
                fp = task.get('file_path')
                if fp:
                    existing_paths.add(str(Path(fp).resolve()))

            # Find new files only
            new_data_files = []
            for data_path in all_data_files:
                normalized = str(data_path.resolve())
                if normalized not in existing_paths:
                    new_data_files.append(data_path)

            new_photometry_files = []
            for fp_path in photometry_files:
                normalized = str(fp_path.resolve())
                if normalized not in existing_paths:
                    new_photometry_files.append(fp_path)

            if not new_data_files and not new_photometry_files:
                total_existing = len(all_data_files)

                # Check for incomplete metadata
                files_needing_metadata = []
                for i, task in enumerate(self.mw._master_file_list):
                    protocol = task.get('protocol', '')
                    if protocol in ('Loading...', '', 'Unknown') or not protocol:
                        file_path = task.get('file_path')
                        if file_path:
                            files_needing_metadata.append((i, Path(file_path)))

                if files_needing_metadata:
                    self.mw._log_status_message(f"Reloading metadata for {len(files_needing_metadata)} files...", 0)
                    self._reload_incomplete_metadata(files_needing_metadata)
                else:
                    self.mw._log_status_message(f"No new files found ({total_existing} files already in list)", 3000)
                    self.mw.scanFilesButton.setEnabled(True)
                return

            # Count by type
            new_abf = len([f for f in new_data_files if f.suffix.lower() == '.abf'])
            new_smrx = len([f for f in new_data_files if f.suffix.lower() == '.smrx'])
            new_edf = len([f for f in new_data_files if f.suffix.lower() == '.edf'])
            new_photometry = len(new_photometry_files)

            # Track starting position
            start_row = len(self.mw._master_file_list)
            new_files_data = []

            # Process data files
            for i, data_path in enumerate(new_data_files):
                file_type = data_path.suffix.lower()[1:]
                path_info = extract_path_keywords(data_path, Path(self.mw._project_directory))

                keywords_display = []
                if path_info.get('relative_path'):
                    keywords_display.append(path_info['relative_path'])
                elif path_info['subdirs']:
                    keywords_display.append('/'.join(path_info['subdirs']))

                if path_info['power_levels']:
                    keywords_display.extend(path_info['power_levels'])
                if path_info['animal_ids']:
                    keywords_display.extend([f"ID:{id}" for id in path_info['animal_ids']])

                power_auto = path_info['power_levels'][0] if path_info['power_levels'] else ''
                animal_id_auto = path_info['animal_ids'][0] if path_info['animal_ids'] else ''

                file_info = {
                    'file_path': data_path,
                    'file_name': data_path.name,
                    'file_type': file_type,
                    'protocol': 'Loading...',
                    'channel_count': 0,
                    'sweep_count': 0,
                    'stim_channels': [],
                    'stim_frequency': '',
                    'path_keywords': path_info,
                    'keywords_display': ', '.join(keywords_display) if keywords_display else '',
                    'channel': '',
                    'stim_channel': '',
                    'events_channel': '',
                    'strain': '',
                    'stim_type': '',
                    'power': power_auto,
                    'sex': '',
                    'animal_id': animal_id_auto,
                    'status': 'pending',
                    'export_path': '',
                    'export_date': '',
                    'export_version': '',
                    'exports': {
                        'npz': False,
                        'timeseries_csv': False,
                        'breaths_csv': False,
                        'events_csv': False,
                        'pdf': False,
                        'session_state': False,
                        'ml_training': False,
                    }
                }
                new_files_data.append(file_info)
                self.mw._master_file_list.append(file_info)

            # Process photometry files
            for fp_path in new_photometry_files:
                fp_info = project_builder.extract_photometry_info(fp_path)

                if fp_info:
                    path_info = extract_path_keywords(fp_path, Path(self.mw._project_directory))

                    keywords_display = []
                    if path_info.get('relative_path'):
                        keywords_display.append(path_info['relative_path'])
                    elif path_info['subdirs']:
                        keywords_display.append('/'.join(path_info['subdirs']))

                    power_auto = path_info['power_levels'][0] if path_info['power_levels'] else ''
                    animal_id_auto = path_info['animal_ids'][0] if path_info['animal_ids'] else ''

                    regions_str = ', '.join(fp_info.get('signal_columns', [])) if fp_info.get('signal_columns') else 'Unknown'
                    channel_info = f"{fp_info.get('region_count', 0)} regions ({regions_str})"

                    file_info = {
                        'file_path': fp_path,
                        'file_name': fp_info.get('file_name', fp_path.name),
                        'file_type': 'photometry',
                        'protocol': fp_info.get('protocol', 'Neurophotometrics'),
                        'channel_count': fp_info.get('region_count', 0),
                        'sweep_count': 1,
                        'stim_channels': [],
                        'stim_frequency': '',
                        'path_keywords': path_info,
                        'keywords_display': ', '.join(keywords_display) if keywords_display else '',
                        'signal_columns': fp_info.get('signal_columns', []),
                        'led_states': fp_info.get('led_states', set()),
                        'led_info': fp_info.get('led_info', 'Unknown'),
                        'has_ai_data': fp_info.get('has_ai_data', False),
                        'ai_data_path': fp_info.get('ai_data_path'),
                        'has_npz': fp_info.get('has_npz', False),
                        'npz_path': fp_info.get('npz_path'),
                        'duration_sec': fp_info.get('duration_sec', 0),
                        'sample_rate': fp_info.get('sample_rate', 0),
                        'channel': channel_info,
                        'stim_channel': '',
                        'events_channel': '',
                        'strain': '',
                        'stim_type': '',
                        'power': power_auto,
                        'sex': '',
                        'animal_id': animal_id_auto,
                        'status': 'pending',
                        'export_path': '',
                        'export_date': '',
                        'export_version': '',
                        'exports': {
                            'npz': fp_info.get('has_npz', False),
                            'timeseries_csv': False,
                            'breaths_csv': False,
                            'events_csv': False,
                            'pdf': False,
                            'session_state': False,
                            'ml_training': False,
                        }
                    }
                    new_files_data.append(file_info)
                    self.mw._master_file_list.append(file_info)
                else:
                    print(f"[scan-manager] Failed to extract metadata from {fp_path}")

            # Upsert new files into SQLite DB
            if new_files_data:
                try:
                    store = self.mw._project_viewmodel.service.store
                    for fi in new_files_data:
                        # Convert Path to string for DB storage
                        db_data = {k: (str(v) if isinstance(v, Path) else v)
                                   for k, v in fi.items()
                                   if k not in ('path_keywords', 'stim_channels',
                                                'exports', 'signal_columns',
                                                'led_states', 'led_info',
                                                'has_ai_data', 'ai_data_path',
                                                'has_npz', 'npz_path',
                                                'duration_sec', 'sample_rate')}
                        eid = store.upsert_experiment(db_data)
                        fi['experiment_id'] = eid
                    print(f"[scan-manager] Upserted {len(new_files_data)} experiments to DB")
                except Exception as e:
                    print(f"[scan-manager] DB upsert error: {e}")

            # Rebuild table
            self.mw._rebuild_table_from_master_list()
            self.mw._auto_fit_table_columns()

            # Update summary
            total_files = len(self.mw._master_file_list)
            type_counts = []
            if new_abf:
                type_counts.append(f"{new_abf} ABF")
            if new_smrx:
                type_counts.append(f"{new_smrx} SMRX")
            if new_edf:
                type_counts.append(f"{new_edf} EDF")
            if new_photometry:
                type_counts.append(f"{new_photometry} Photometry")
            type_summary = ", ".join(type_counts) if type_counts else "0 files"

            total_new = len(new_data_files) + len(new_photometry_files)
            if new_data_files:
                summary_text = f"Summary: {total_files} files total | Added {type_summary} | Loading metadata..."
                self.mw._log_status_message(f"Added {total_new} new files, loading metadata...", 3000)
            else:
                summary_text = f"Summary: {total_files} files total | Added {type_summary}"
                self.mw._log_status_message(f"Added {total_new} new files", 3000)
                self.mw.scanFilesButton.setEnabled(True)
            self.mw.summaryLabel.setText(summary_text)

            # Store reference to new files data
            self.mw._discovered_files_data = new_files_data

            # PHASE 3: Load metadata in background
            if new_data_files:
                self._start_background_metadata_loading(new_data_files, start_row_offset=start_row)

        except Exception as e:
            if progress:
                progress.close()
            self.mw.scanFilesButton.setEnabled(True)
            self.mw._show_error("Scan Error", f"Failed to scan directory:\n{e}")
            print(f"[scan-manager] Error: {e}")
            import traceback
            traceback.print_exc()

    def _start_background_metadata_loading(self, data_files: List[Path], start_row_offset: int = 0):
        """Start background thread to load metadata using parallel processing."""
        self._metadata_row_offset = start_row_offset

        # Show progress bar
        self.mw.projectProgressBar.setVisible(True)
        self.mw.projectProgressBar.setValue(0)
        self.mw.projectProgressBar.setFormat(f"Loading metadata: 0/{len(data_files)} (0%)")

        self._metadata_thread = MetadataThread(data_files)
        self._metadata_thread.batch_progress.connect(self._update_file_metadata_batch)
        self._metadata_thread.finished.connect(self._metadata_finished)
        self._metadata_thread.start()
        print(f"[scan-manager] Started background loading for {len(data_files)} files (offset={start_row_offset})")

    def _update_file_metadata_batch(self, batch: list, total: int, completed: int):
        """Update table cells with a batch of loaded metadata."""
        row_offset = self._metadata_row_offset

        for index, metadata in batch:
            if metadata:
                # Update discovered files data
                if index < len(self.mw._discovered_files_data):
                    self.mw._discovered_files_data[index]['protocol'] = metadata['protocol']
                    self.mw._discovered_files_data[index]['channel_count'] = metadata.get('channel_count', 0)
                    self.mw._discovered_files_data[index]['sweep_count'] = metadata.get('sweep_count', 0)
                    self.mw._discovered_files_data[index]['stim_channels'] = metadata.get('stim_channels', [])
                    self.mw._discovered_files_data[index]['stim_frequency'] = metadata.get('stim_frequency', '')
                    if metadata.get('stim_frequency') and not self.mw._discovered_files_data[index].get('stim_type'):
                        self.mw._discovered_files_data[index]['stim_type'] = metadata.get('stim_frequency')

                # Update master list
                master_idx = row_offset + index
                if master_idx < len(self.mw._master_file_list):
                    task = self.mw._master_file_list[master_idx]
                    task['protocol'] = metadata['protocol']
                    task['channel_count'] = metadata.get('channel_count', 0)
                    task['sweep_count'] = metadata.get('sweep_count', 0)
                    task['stim_channels'] = metadata.get('stim_channels', [])
                    task['stim_frequency'] = metadata.get('stim_frequency', '')
                    if metadata.get('stim_frequency') and not task.get('stim_type'):
                        task['stim_type'] = metadata.get('stim_frequency')
                    if metadata.get('stim_channels') and not task.get('events_channel'):
                        task['events_channel'] = ', '.join(metadata.get('stim_channels', []))

                    if master_idx < self.mw._file_table_model.rowCount():
                        self.mw._file_table_model.update_row(master_idx, task)

        # Update progress bar
        if batch:
            progress_pct = int(completed / total * 100)
            self.mw.projectProgressBar.setValue(progress_pct)
            self.mw.projectProgressBar.setFormat(f"Loading metadata: {completed}/{total} ({progress_pct}%)")
            self.mw._log_status_message(f"Loading metadata... {completed}/{total}", 0)

    def _metadata_finished(self, protocols: set):
        """Called when background loading completes."""
        self.mw._auto_fit_table_columns()
        self.mw._apply_all_row_styling()

        # Hide progress bar
        self.mw.projectProgressBar.setVisible(False)
        self.mw.projectProgressBar.setValue(0)

        total_files = len(self.mw._master_file_list)
        new_files = len(self.mw._discovered_files_data)

        summary_text = f"Summary: {total_files} ABF files | {len(protocols)} protocols"
        self.mw.summaryLabel.setText(summary_text)
        self.mw._log_status_message(f"Loaded metadata for {new_files} files ({total_files} total)", 3000)
        print(f"[scan-manager] Complete! {len(protocols)} protocols: {sorted(protocols)}")

        self.mw.scanFilesButton.setEnabled(True)

        if hasattr(self.mw, 'tableFilterEdit') and self.mw.tableFilterEdit.text():
            self.mw._on_table_filter_changed()

        self._metadata_row_offset = 0
        self.mw._project_builder.project_autosave()
        self.mw._populate_consolidation_source_list()

    def _reload_incomplete_metadata(self, files_with_indices: list):
        """Reload metadata for files that have incomplete data."""
        if not files_with_indices:
            return

        file_paths = [fp for _, fp in files_with_indices]
        row_indices = [idx for idx, _ in files_with_indices]

        def on_reload_batch(batch, total, completed):
            for orig_idx, row_idx, metadata in batch:
                if metadata and row_idx < len(self.mw._master_file_list):
                    task = self.mw._master_file_list[row_idx]
                    task['protocol'] = metadata.get('protocol', 'Unknown')
                    task['channel_count'] = metadata.get('channel_count', 0)
                    task['sweep_count'] = metadata.get('sweep_count', 0)
                    task['stim_channels'] = metadata.get('stim_channels', [])
                    task['stim_frequency'] = metadata.get('stim_frequency', '')

                    if row_idx < self.mw._file_table_model.rowCount():
                        self.mw._file_table_model.update_row(row_idx, task)

            progress_pct = int(completed / total * 100)
            self.mw.projectProgressBar.setValue(progress_pct)
            self.mw.projectProgressBar.setFormat(f"Reloading metadata: {completed}/{total} ({progress_pct}%)")

        def on_reload_finished(protocols):
            self.mw.projectProgressBar.setVisible(False)
            self.mw.scanFilesButton.setEnabled(True)
            self.mw._log_status_message(f"Reloaded metadata ({len(protocols)} protocols found)", 3000)
            self.mw._project_builder.project_autosave()

        # Show progress bar
        self.mw.projectProgressBar.setVisible(True)
        self.mw.projectProgressBar.setValue(0)
        self.mw.projectProgressBar.setFormat(f"Reloading metadata: 0/{len(file_paths)} (0%)")

        self._reload_thread = ReloadMetadataThread(file_paths, row_indices)
        self._reload_thread.batch_progress.connect(on_reload_batch)
        self._reload_thread.finished.connect(on_reload_finished)
        self._reload_thread.start()

    def on_project_scan_saved_data(self, scan_folder: Path = None, silent: bool = False):
        """Scan for existing saved data files and auto-populate the table."""
        import numpy as np
        import json

        if not self.mw._master_file_list:
            if not silent:
                self.mw._show_warning("No Files", "Please scan for ABF files first.")
            return

        # Show progress dialog
        progress = None
        if not silent:
            progress = QProgressDialog("Scanning for saved data...", "Cancel", 0, 100, self.mw)
            progress.setWindowTitle("Scanning Saved Data")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.show()
            QApplication.processEvents()

        # Find analysis folders
        if scan_folder:
            analysis_folders = [scan_folder] if scan_folder.exists() else []
            print(f"[scan-manager] Focused scan of: {scan_folder}")
        else:
            base_dir = Path(self.mw._project_directory) if self.mw._project_directory else Path.cwd()
            analysis_folders = list(base_dir.glob("**/Pleth_App_analysis"))

        if not analysis_folders:
            if progress:
                progress.close()
            if not silent:
                self.mw._show_info("No Saved Data", "No 'Pleth_App_analysis' folders found.\n\nAnalyzed data is saved to this folder.")
            return

        # Build mapping of ABF names to saved data
        saved_data_map = {}

        if progress:
            progress.setLabelText(f"Scanning {len(analysis_folders)} analysis folders...")
            progress.setValue(10)
            QApplication.processEvents()

        # Collect ALL NPZ files, sort by modification time (newest first)
        all_npz_files = []
        for folder in analysis_folders:
            for npz_file in folder.glob("*_bundle.npz"):
                try:
                    mtime = npz_file.stat().st_mtime
                except:
                    mtime = 0
                all_npz_files.append((npz_file, folder, mtime))

        all_npz_files.sort(key=lambda x: x[2], reverse=True)
        print(f"[scan-manager] Found {len(all_npz_files)} NPZ bundle files, processing newest first")

        for npz_file, folder, mtime in all_npz_files:
            stem = npz_file.stem.replace('_bundle', '')

            # Read metadata from NPZ
            npz_metadata = None
            source_file_from_npz = None
            channel_from_npz = None

            try:
                with np.load(npz_file, allow_pickle=True) as data:
                    if 'meta_json' in data:
                        meta_str = str(data['meta_json'])
                        npz_metadata = json.loads(meta_str)
                        if isinstance(npz_metadata, dict):
                            source_file_from_npz = npz_metadata.get('abf_path', '')
                            channel_from_npz = npz_metadata.get('analyze_channel', '')
                    elif 'source_file' in data:
                        source_file_from_npz = str(data['source_file'])
                        npz_metadata = {'source_file': source_file_from_npz}
            except Exception as e:
                print(f"[scan-manager] Could not read metadata from {npz_file.name}: {e}")

            # Match to ABF file
            matched_task = None
            match_method = None

            if source_file_from_npz:
                source_stem = Path(source_file_from_npz).stem
                for task in self.mw._master_file_list:
                    if task.get('is_sub_row'):
                        continue
                    abf_path = Path(task.get('file_path', ''))
                    if abf_path.stem == source_stem:
                        matched_task = task
                        match_method = "NPZ metadata"
                        break

            if not matched_task:
                for task in self.mw._master_file_list:
                    if task.get('is_sub_row'):
                        continue
                    abf_path = Path(task.get('file_path', ''))
                    abf_name = abf_path.stem
                    if not abf_name:
                        continue

                    pattern = r'(^|_)' + re.escape(abf_name) + r'(_|$)'
                    if re.search(pattern, stem):
                        matched_task = task
                        match_method = "pattern"
                        break

            if matched_task:
                abf_name = Path(matched_task.get('file_path', '')).stem
                print(f"[scan-manager] Matched {npz_file.name} to {abf_name} via {match_method}")

                key = str(matched_task.get('file_path'))
                channel = channel_from_npz or ''
                if not channel:
                    channel_match = re.search(r'_(AD\d+)_', stem) or re.search(r'_(AD\d+)$', stem)
                    channel = channel_match.group(1) if channel_match else ''

                if key not in saved_data_map:
                    saved_data_map[key] = {}

                if channel in saved_data_map[key]:
                    if 'older_npz_files' not in saved_data_map[key][channel]:
                        saved_data_map[key][channel]['older_npz_files'] = []
                    saved_data_map[key][channel]['older_npz_files'].append({
                        'file': str(npz_file),
                        'date': datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M'),
                    })
                    continue

                export_info = {
                    'export_path': str(folder),
                    'export_date': datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M'),
                    'npz': True,
                    'timeseries_csv': (folder / f"{stem}_means_by_time.csv").exists(),
                    'breaths_csv': (folder / f"{stem}_breaths.csv").exists(),
                    'events_csv': (folder / f"{stem}_events.csv").exists(),
                    'pdf': (folder / f"{stem}_summary.pdf").exists(),
                    'session_state': (folder / f"{stem}_session.npz").exists(),
                    'ml_training': False,
                    'npz_metadata': npz_metadata,
                }

                if npz_metadata:
                    ui_meta = npz_metadata.get('ui_meta', {})
                    if isinstance(ui_meta, dict):
                        export_info['strain'] = ui_meta.get('strain', '')
                        export_info['stim_type'] = ui_meta.get('stim', '')
                        export_info['power'] = ui_meta.get('power', '')
                        export_info['sex'] = ui_meta.get('sex', '')
                        export_info['animal_id'] = ui_meta.get('animal', '')
                        export_info['stim_channel'] = npz_metadata.get('stim_chan', '')
                        export_info['events_channel'] = npz_metadata.get('event_channel', '')

                saved_data_map[key][channel] = export_info

        if progress:
            progress.setValue(50)
            QApplication.processEvents()

        if not saved_data_map:
            if progress:
                progress.close()
            if not silent:
                self.mw._show_info("No Matches", "No saved data files matched the current ABF files.")
            return

        # Update master list with found saved data
        updated_count = 0
        new_rows_created = 0

        existing_file_channels = set()
        for task in self.mw._master_file_list:
            fp = str(task.get('file_path', ''))
            ch = task.get('channel', '')
            if fp and ch:
                existing_file_channels.add((fp, ch))

        # First pass: update existing sub-rows
        for row, task in enumerate(self.mw._master_file_list):
            file_key = str(task.get('file_path'))
            if file_key not in saved_data_map:
                continue

            existing_channel = task.get('channel', '')
            is_sub_row = task.get('is_sub_row', False)
            export_data = saved_data_map[file_key]

            if is_sub_row and existing_channel and existing_channel in export_data:
                info = export_data[existing_channel]
                self.mw._update_task_with_export_info(task, info, row)
                updated_count += 1

            if progress and row % 10 == 0:
                progress.setValue(50 + int(25 * row / len(self.mw._master_file_list)))
                QApplication.processEvents()

        # Second pass: create new sub-rows
        rows_to_add = []
        for file_key, export_data in saved_data_map.items():
            for channel, info in export_data.items():
                if (file_key, channel) not in existing_file_channels:
                    rows_to_add.append((file_key, channel, info))
                    existing_file_channels.add((file_key, channel))

        for file_key, channel, info in rows_to_add:
            parent_task = None
            parent_row = None
            for row, task in enumerate(self.mw._master_file_list):
                if str(task.get('file_path')) == file_key and not task.get('is_sub_row', False):
                    parent_task = task
                    parent_row = row
                    break

            if not parent_task:
                for row, task in enumerate(self.mw._master_file_list):
                    if str(task.get('file_path')) == file_key:
                        parent_task = task
                        parent_row = row
                        break

            if parent_task:
                self.mw._create_sub_row_from_saved_data(parent_task, parent_row, channel, info)
                new_rows_created += 1
                updated_count += 1

        if progress:
            progress.setValue(90)
            QApplication.processEvents()

        self.mw._rebuild_table_from_master_list()

        if progress:
            progress.close()

        # Count warnings
        warnings_count = 0
        conflicts_count = 0
        older_files_count = 0
        for task in self.mw._master_file_list:
            warnings = task.get('scan_warnings', {})
            if warnings:
                warnings_count += 1
                if warnings.get('conflicts'):
                    conflicts_count += 1
                if warnings.get('older_npz_count', 0) > 0:
                    older_files_count += 1

        msg = f"Found saved data for {updated_count} analyses"
        if new_rows_created > 0:
            msg += f" ({new_rows_created} new rows created)"
        if warnings_count > 0:
            warning_details = []
            if conflicts_count > 0:
                warning_details.append(f"{conflicts_count} conflicts")
            if older_files_count > 0:
                warning_details.append(f"{older_files_count} with multiple NPZ files")
            msg += f" - {', '.join(warning_details)}"
        self.mw._log_status_message(msg, 5000 if warnings_count > 0 else 3000)

        if hasattr(self.mw, 'resolveConflictsButton'):
            self.mw.resolveConflictsButton.setEnabled(warnings_count > 0)
            if warnings_count > 0:
                self.mw.resolveConflictsButton.setText(f"Resolve Conflicts ({warnings_count})")
            else:
                self.mw.resolveConflictsButton.setText("Resolve Conflicts")
