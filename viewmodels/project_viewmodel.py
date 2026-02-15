"""
Project view model — Qt integration for the project service.

Provides QObject signals for UI binding and commands for user actions.
Acts as the bridge between the UI (ProjectBuilderManager) and the
service layer (ProjectService).
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from core.services.project_service import ProjectService


class ProjectViewModel(QObject):
    """
    ViewModel for the project / master file list.

    Signals:
        file_updated(str, dict)      — (file_path, updated_fields)
        files_added(list)            — [file_entry, ...]
        files_removed(list)          — [file_path, ...]
        scan_progress(int, int, str) — (current, total, message)
        scan_finished(int)           — number of new files found
        metadata_changed()           — general refresh trigger
        project_loaded(dict)         — project summary
        project_saved(str)           — path to saved file
        completeness_changed(dict)   — completeness stats
        external_update(list)        — list of changed file_paths from file watcher
    """

    # Signals
    file_updated = pyqtSignal(str, dict)
    files_added = pyqtSignal(list)
    files_removed = pyqtSignal(list)
    scan_progress = pyqtSignal(int, int, str)
    scan_finished = pyqtSignal(int)
    metadata_changed = pyqtSignal()
    project_loaded = pyqtSignal(dict)
    project_saved = pyqtSignal(str)
    completeness_changed = pyqtSignal(dict)
    external_update = pyqtSignal(list)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)

        self._service = ProjectService()

        # File watcher state
        self._watcher = None
        self._poll_timer = None  # For network drive polling fallback
        self._last_mtime = 0.0  # Track project file mtime for polling
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(200)
        self._debounce_timer.timeout.connect(self._on_file_change_debounced)
        self._saving = False  # Suppress watcher while we save

    @property
    def service(self) -> ProjectService:
        """Access the underlying service."""
        return self._service

    # ------------------------------------------------------------------
    # Project operations
    # ------------------------------------------------------------------

    def open_project(self, folder: str) -> Dict[str, Any]:
        """Open or create a project."""
        result = self._service.open_project(Path(folder))
        self._setup_file_watcher()
        self.project_loaded.emit(result)
        self.metadata_changed.emit()
        return result

    def load_project(self, project_path: str) -> Dict[str, Any]:
        """Load a project from file."""
        result = self._service.load_project(Path(project_path))
        self._setup_file_watcher()
        self.project_loaded.emit(result)
        self.metadata_changed.emit()
        return result

    def save_project(self, name: Optional[str] = None) -> str:
        """Save the current project."""
        self._saving = True
        try:
            path = self._service.save_project(name)
            self.project_saved.emit(path)
            # Re-setup watcher in case path changed
            self._setup_file_watcher()
            return path
        finally:
            # Re-enable watcher after a short delay (OS may fire change event)
            QTimer.singleShot(500, self._clear_saving_flag)

    def _clear_saving_flag(self):
        self._saving = False

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan_folder(
        self,
        root: Optional[str] = None,
        recursive: bool = True,
        file_types: Optional[List[str]] = None,
        auto_add: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Scan for new files.

        Args:
            root: Folder to scan (None = project data_directory).
            recursive: Search subdirectories.
            file_types: Types to scan.
            auto_add: If True, automatically add discovered files to project.

        Returns:
            List of newly discovered file entries.
        """
        def progress(current, total, msg):
            self.scan_progress.emit(current, total, msg)

        root_path = Path(root) if root else None
        new_files = self._service.scan_folder(
            root=root_path,
            recursive=recursive,
            file_types=file_types,
            progress_callback=progress,
        )

        if auto_add and new_files:
            count = self._service.add_files(new_files)
            self.files_added.emit(new_files)
            self.scan_finished.emit(count)
            self.metadata_changed.emit()
        else:
            self.scan_finished.emit(0)

        return new_files

    def load_file_metadata(self) -> int:
        """Load detailed metadata for all files missing it."""
        def progress(current, total, msg):
            self.scan_progress.emit(current, total, msg)

        updated = self._service.load_file_metadata(progress_callback=progress)
        if updated > 0:
            self.metadata_changed.emit()
        return updated

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def update_file_metadata(self, file_path: str, updates: Dict[str, Any]) -> Optional[Dict]:
        """Update metadata for a single file."""
        result = self._service.update_file_metadata(file_path, updates)
        if result:
            self.file_updated.emit(file_path, updates)
            self.completeness_changed.emit(self._service.get_metadata_completeness())
        return result

    def batch_update(self, file_paths: List[str], updates: Dict[str, Any]) -> int:
        """Update multiple files with the same metadata."""
        count = self._service.batch_update(file_paths, updates)
        if count > 0:
            for fp in file_paths:
                self.file_updated.emit(fp, updates)
            self.metadata_changed.emit()
            self.completeness_changed.emit(self._service.get_metadata_completeness())
        return count

    def add_files(self, file_entries: List[Dict[str, Any]]) -> int:
        """Add new file entries."""
        count = self._service.add_files(file_entries)
        if count > 0:
            self.files_added.emit(file_entries)
            self.metadata_changed.emit()
        return count

    def remove_file(self, file_path: str) -> bool:
        """Remove a file from the project."""
        result = self._service.remove_file(file_path)
        if result:
            self.files_removed.emit([file_path])
            self.metadata_changed.emit()
        return result

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_project_files(self) -> List[Dict[str, Any]]:
        """Get all files with metadata."""
        return self._service.get_project_files()

    def get_metadata_completeness(self) -> Dict[str, Any]:
        """Get completeness stats."""
        return self._service.get_metadata_completeness()

    def get_unique_values(self, field: str) -> List[str]:
        """Get unique values for a field."""
        return self._service.get_unique_values(field)

    @property
    def file_count(self) -> int:
        return self._service.get_file_count()

    @property
    def is_dirty(self) -> bool:
        return self._service.is_dirty

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------

    def read_notes_file(self, path: str) -> List[Dict]:
        """Read and parse a notes file."""
        return self._service.read_notes_file(Path(path))

    def match_notes_to_files(self, notes_entries: List[Dict]) -> List[Dict]:
        """Match notes entries to recording files."""
        return self._service.match_notes_to_files(notes_entries)

    def discover_notes_files(self, root: Optional[str] = None) -> List[Dict]:
        """Discover notes files in project folder."""
        return self._service.discover_notes_files(Path(root) if root else None)

    # ------------------------------------------------------------------
    # Pattern application
    # ------------------------------------------------------------------

    def apply_patterns(self) -> Dict[str, Any]:
        """Apply cached patterns to unfilled fields."""
        result = self._service.apply_patterns()
        if result.get("applied", 0) > 0:
            self.metadata_changed.emit()
            self.completeness_changed.emit(self._service.get_metadata_completeness())
        return result

    # ------------------------------------------------------------------
    # Subrows
    # ------------------------------------------------------------------

    def add_subrow(self, file_path: str, channel: str, **kwargs) -> Optional[Dict]:
        """Add a subrow to a multi-animal recording."""
        result = self._service.add_subrow(file_path, channel, **kwargs)
        if result:
            self.file_updated.emit(file_path, {"subrows": "added"})
        return result

    def get_subrows(self, file_path: str) -> List[Dict]:
        """Get subrows for a file."""
        return self._service.get_subrows(file_path)

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def get_provenance(self, file_path: str, field: Optional[str] = None) -> List[Dict]:
        """Get provenance records for a file."""
        return self._service.get_provenance(file_path, field)

    # ------------------------------------------------------------------
    # File labels (ML training data)
    # ------------------------------------------------------------------

    def label_file(self, file_path: str, label: str) -> None:
        """Label a file's type for ML classifier training."""
        if self._service.cache:
            self._service.cache.save_file_label(file_path, label)

    # ------------------------------------------------------------------
    # Grouped view
    # ------------------------------------------------------------------

    def get_files_grouped(self) -> Dict[str, Any]:
        """Get files grouped by folder with summary stats."""
        return self._service.get_files_grouped()

    # ------------------------------------------------------------------
    # File watcher (Phase 3 — real-time UI updates)
    # ------------------------------------------------------------------

    def _setup_file_watcher(self):
        """Set up QFileSystemWatcher on the project file.

        Detects network paths (UNC or mapped drives) and falls back to
        mtime polling when QFileSystemWatcher may not fire reliably.
        """
        from PyQt6.QtCore import QFileSystemWatcher

        # Clean up old watcher + poll timer
        if self._watcher is not None:
            self._watcher.fileChanged.disconnect(self._on_file_changed)
            self._watcher.deleteLater()
            self._watcher = None
        if self._poll_timer is not None:
            self._poll_timer.stop()
            self._poll_timer = None

        project_path = self._service.project_path
        if project_path is None or not project_path.exists():
            return

        # Detect network path — UNC (\\server\share) or mapped drive
        path_str = str(project_path)
        is_network = path_str.startswith("\\\\") or self._is_mapped_drive(path_str)

        if is_network:
            # Network drive: use polling fallback (5-second interval)
            self._start_polling_watcher(project_path)
        else:
            # Local drive: use QFileSystemWatcher (instant)
            self._watcher = QFileSystemWatcher([path_str], self)
            self._watcher.fileChanged.connect(self._on_file_changed)

    @staticmethod
    def _is_mapped_drive(path_str: str) -> bool:
        """Check if a drive letter is a mapped network drive."""
        import os
        if len(path_str) >= 2 and path_str[1] == ':':
            drive = path_str[0].upper()
            # Check common network drive letters (Z:, Y:, X:, etc.)
            # More reliable: use win32api if available
            try:
                import ctypes
                drive_type = ctypes.windll.kernel32.GetDriveTypeW(f"{drive}:\\")
                return drive_type == 4  # DRIVE_REMOTE
            except (ImportError, AttributeError, OSError):
                pass
        return False

    def _start_polling_watcher(self, project_path: Path):
        """Start mtime-based polling for network drive projects."""
        try:
            self._last_mtime = project_path.stat().st_mtime
        except OSError:
            self._last_mtime = 0.0

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(5000)  # 5 seconds
        self._poll_timer.timeout.connect(self._poll_check)
        self._poll_timer.start()

    def _poll_check(self):
        """Check if project file mtime has changed (polling fallback)."""
        if self._saving:
            return
        project_path = self._service.project_path
        if project_path is None or not project_path.exists():
            return
        try:
            mtime = project_path.stat().st_mtime
            if mtime != self._last_mtime:
                self._last_mtime = mtime
                self._debounce_timer.start()
        except OSError:
            pass

    def _on_file_changed(self, path: str):
        """Handle file change notification (debounced)."""
        if self._saving:
            return  # Ignore changes we caused

        # Restart debounce timer
        self._debounce_timer.start()

    def _on_file_change_debounced(self):
        """Process debounced file change — diff and emit updates."""
        if self._saving:
            return

        project_path = self._service.project_path
        if project_path is None or not project_path.exists():
            return

        # Reload from disk and diff
        old_files = {
            str(Path(f.get("file_path", "")).resolve()): dict(f)
            for f in self._service._files
        }

        try:
            self._service.load_project(project_path)
        except Exception as e:
            print(f"[project-viewmodel] Error reloading project: {e}")
            return

        # Find changed files
        changed_paths = []
        for f in self._service._files:
            fp = str(Path(f.get("file_path", "")).resolve())
            old = old_files.get(fp)
            if old is None:
                changed_paths.append(f.get("file_path", ""))
            else:
                # Check if any field changed
                for key in f:
                    if f.get(key) != old.get(key):
                        changed_paths.append(f.get("file_path", ""))
                        break

        if changed_paths:
            self.external_update.emit(changed_paths)
            self.metadata_changed.emit()

        # Re-add file to watcher (Qt removes it after change on some platforms)
        if self._watcher and project_path.exists():
            watched = self._watcher.files()
            if str(project_path) not in watched:
                self._watcher.addPath(str(project_path))

    def reload_from_disk(self):
        """Force reload the project from disk."""
        project_path = self._service.project_path
        if project_path and project_path.exists():
            self._saving = True  # Suppress watcher during reload
            try:
                self._service.load_project(project_path)
                self.metadata_changed.emit()
            finally:
                QTimer.singleShot(500, self._clear_saving_flag)
