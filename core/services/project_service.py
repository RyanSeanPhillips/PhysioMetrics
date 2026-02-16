"""
Project service — pure Python, no PyQt6 imports.

v3: Uses ExperimentStore (v2 schema) with flat experiments table.
Snapshots replace projects. No notes parsing (Claude reads notes natively).
Merges former ProjectManager JSON I/O into this service.
"""

import json
import hashlib
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


class ProjectService:
    """
    Pure-Python service for experiment metadata operations.

    Uses ExperimentStore (v2 schema) as primary store.
    Snapshots replace projects — a snapshot is a named filter query
    that can be exported to .physiometrics JSON for sharing.
    """

    def __init__(self):
        self._store = None  # Lazy-loaded ExperimentStore
        self._snapshot_id: Optional[int] = None
        self._snapshot_name: str = ""
        self._data_directory: Optional[Path] = None
        self._project_path: Optional[Path] = None  # .physiometrics JSON path
        self._dirty = False
        self._last_merge_report: Optional[Dict] = None

        # Legacy compatibility: in-memory file list (populated from DB)
        self._files: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Store (lazy-loaded)
    # ------------------------------------------------------------------

    @property
    def store(self):
        """Lazy-load the ExperimentStore."""
        if self._store is None:
            from core.adapters.experiment_store import ExperimentStore
            self._store = ExperimentStore()
        return self._store

    # ------------------------------------------------------------------
    # Open / save
    # ------------------------------------------------------------------

    def open_project(self, folder: Path) -> Dict[str, Any]:
        """Open or create a snapshot for a folder.

        1. Find/create snapshot by data_directory
        2. If .physiometrics JSON exists, import experiments from it
        3. Populate in-memory _files from DB

        Returns summary dict.
        """
        folder = Path(folder)
        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder}")

        self._data_directory = folder
        norm_dir = str(folder.resolve())

        # Find existing snapshot for this directory
        snap = self.store.get_snapshot_by_directory(norm_dir)
        if snap:
            self._snapshot_id = snap["snapshot_id"]
            self._snapshot_name = snap["name"]
        else:
            self._snapshot_name = folder.name
            self._snapshot_id = self.store.create_snapshot(
                name=self._snapshot_name,
                data_directory=norm_dir,
            )

        # Look for .physiometrics file
        pm_files = list(folder.glob("*.physiometrics"))
        json_path = None
        if pm_files:
            pm_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            json_path = pm_files[0]
        self._project_path = json_path

        # Import from JSON if exists
        merge_report = None
        if json_path:
            merge_report = self._sync_json(json_path)

        # Populate in-memory file list
        self._refresh_files_from_db()
        self._dirty = False
        self._last_merge_report = merge_report

        result = {
            "project_name": self._snapshot_name,
            "data_directory": str(folder),
            "file_count": len(self._files),
            "status": "loaded" if json_path else "created_new",
        }
        if merge_report:
            result["merge_report"] = merge_report
        return result

    def load_project(self, project_path: Path) -> Dict[str, Any]:
        """Load from a .physiometrics file (legacy compat)."""
        return self.open_project(Path(project_path).parent)

    def save_project(self, name: Optional[str] = None) -> str:
        """Save current snapshot to .physiometrics JSON on disk.

        Returns path to saved file.
        """
        if self._data_directory is None:
            raise RuntimeError("No project open — call open_project() first")

        project_name = name or self._snapshot_name or self._data_directory.name
        self._snapshot_name = project_name

        # Update snapshot name
        if self._snapshot_id:
            self.store.create_snapshot(
                name=project_name,
                data_directory=str(self._data_directory.resolve()),
            )

        # Export experiments to JSON
        experiments, _ = self.store.get_experiments()
        files_json = []
        for exp in experiments:
            entry = dict(exp)
            entry.pop('experiment_id', None)
            entry.pop('created_at', None)
            entry.pop('updated_at', None)
            # Convert to relative paths
            fp = entry.get('file_path', '')
            if fp and self._data_directory:
                try:
                    rel = Path(fp).relative_to(self._data_directory)
                    entry['file_path'] = str(rel).replace("\\", "/")
                except (ValueError, TypeError):
                    pass
            files_json.append(entry)

        project_data = {
            "version": 3,
            "project_name": project_name,
            "data_directory": ".",
            "created": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "file_count": len(files_json),
            "files": files_json,
            "experiments": [],
            "notes_files": [],
            "custom_columns": self.store.get_dynamic_columns(),
        }

        # Write JSON atomically
        project_filename = f"{self._sanitize_filename(project_name)}.physiometrics"
        project_path = self._data_directory / project_filename

        try:
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.tmp', prefix='physiometrics_', dir=self._data_directory,
            )
            try:
                with os.fdopen(temp_fd, 'w') as f:
                    json.dump(project_data, f, indent=2)
                Path(temp_path).replace(project_path)
            except Exception:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            raise RuntimeError(f"Failed to save project: {e}")

        self._project_path = project_path

        # Update sync metadata
        if project_path.exists() and self._snapshot_id:
            mtime = project_path.stat().st_mtime
            fhash = self.file_hash(project_path)
            self.store.update_snapshot_json_sync(
                self._snapshot_id, str(project_path), mtime, fhash,
            )

        self._dirty = False
        return str(project_path)

    def _sync_json(self, json_path: Path) -> Optional[Dict[str, Any]]:
        """Import/merge a .physiometrics JSON into the DB.

        Compares mtime/hash to skip unnecessary re-imports.
        """
        if not json_path.exists():
            return None

        snap = self.store.get_snapshot(self._snapshot_id) if self._snapshot_id else None

        try:
            current_mtime = json_path.stat().st_mtime
            current_hash = self.file_hash(json_path)
        except OSError:
            return None

        # Skip if already synced
        if snap and snap.get("json_hash") == current_hash and snap.get("json_mtime") == current_mtime:
            return None

        # Backup before merge
        try:
            self.store.backup(trigger_event="merge")
        except Exception as e:
            print(f"[project-service] Backup before merge failed: {e}")

        # Load JSON
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
        except Exception as e:
            print(f"[project-service] Error reading JSON: {e}")
            return None

        # Import
        data_dir = str(self._data_directory) if self._data_directory else ''
        report = self.store.import_from_json(json_data, data_directory=data_dir)

        # Update sync metadata
        if self._snapshot_id:
            self.store.update_snapshot_json_sync(
                self._snapshot_id, str(json_path), current_mtime, current_hash,
            )

        return report

    def sync_json(self) -> Optional[Dict[str, Any]]:
        """Public method to trigger JSON sync (called by ViewModel on file change)."""
        if self._project_path and self._project_path.exists():
            report = self._sync_json(self._project_path)
            if report:
                self._refresh_files_from_db()
            return report
        return None

    def _refresh_files_from_db(self):
        """Reload in-memory _files list from DB."""
        experiments, _ = self.store.get_experiments()
        self._files = experiments

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @property
    def project_path(self) -> Optional[Path]:
        return self._project_path

    @property
    def data_directory(self) -> Optional[Path]:
        return self._data_directory

    @property
    def snapshot_id(self) -> Optional[int]:
        return self._snapshot_id

    # Legacy compat alias
    @property
    def project_id(self) -> Optional[int]:
        return self._snapshot_id

    @property
    def last_merge_report(self) -> Optional[Dict]:
        return self._last_merge_report

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _to_relative(self, abs_path: str) -> str:
        """Convert an absolute path to relative from data_directory."""
        if not self._data_directory or not abs_path:
            return abs_path
        try:
            rel = Path(abs_path).relative_to(self._data_directory)
            return str(rel).replace("\\", "/")
        except (ValueError, TypeError):
            return abs_path

    def _to_absolute(self, rel_or_abs: str) -> str:
        """Resolve a relative-or-absolute path against data_directory."""
        if not rel_or_abs:
            return rel_or_abs
        p = Path(rel_or_abs)
        if p.is_absolute():
            return str(p)
        if self._data_directory:
            return str(self._data_directory / p)
        return rel_or_abs

    def _resolve_experiment_id(self, file_path: str, channel: str = '', animal_id: str = '') -> Optional[int]:
        """Resolve file_path (+ optional channel/animal_id) to experiment_id."""
        # Try as-is first
        exp = self.store.get_experiment_by_key(file_path, channel, animal_id)
        if exp:
            return exp['experiment_id']

        # Try absolute
        abs_path = self._to_absolute(file_path)
        exp = self.store.get_experiment_by_key(abs_path, channel, animal_id)
        if exp:
            return exp['experiment_id']

        # Try all experiments for this file (any channel/animal)
        if not channel and not animal_id:
            exps = self.store.get_experiments_by_file(file_path)
            if not exps:
                exps = self.store.get_experiments_by_file(abs_path)
            if exps:
                return exps[0]['experiment_id']

        return None

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Sanitize a string for use as a filename."""
        import re
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
        return sanitized.strip('. ')[:100] or 'project'

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan_folder(
        self,
        root: Optional[Path] = None,
        recursive: bool = True,
        file_types: Optional[List[str]] = None,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """Discover recording files in a folder tree.

        Returns list of newly discovered file dicts.
        """
        from core import project_builder
        from core.fast_abf_reader import extract_path_keywords

        root = Path(root) if root else self._data_directory
        if root is None:
            raise RuntimeError("No folder to scan")

        files = project_builder.discover_files(
            str(root), recursive=recursive, file_types=file_types,
        )

        abf_files = files.get("abf_files", [])
        smrx_files = files.get("smrx_files", [])
        edf_files = files.get("edf_files", [])
        photometry_files = files.get("photometry_files", [])

        all_data_files = abf_files + smrx_files + edf_files

        # Build set of existing paths from DB
        existing_paths = set()
        existing_exps, _ = self.store.get_experiments()
        for exp in existing_exps:
            fp = exp.get("file_path", "")
            if fp:
                existing_paths.add(str(Path(fp).resolve()))

        new_entries = []
        total = len(all_data_files) + len(photometry_files)
        current = 0

        for data_path in all_data_files:
            normalized = str(data_path.resolve())
            if normalized in existing_paths:
                current += 1
                continue

            file_type = data_path.suffix.lower().lstrip(".")
            path_info = extract_path_keywords(data_path, root)

            keywords_display = []
            if path_info.get("relative_path"):
                keywords_display.append(path_info["relative_path"])
            elif path_info["subdirs"]:
                keywords_display.append("/".join(path_info["subdirs"]))
            if path_info["power_levels"]:
                keywords_display.extend(path_info["power_levels"])
            if path_info["animal_ids"]:
                keywords_display.extend([f"ID:{aid}" for aid in path_info["animal_ids"]])

            entry = {
                "file_path": str(data_path),
                "file_name": data_path.name,
                "file_type": file_type,
                "protocol": "",
                "channel_count": 0,
                "sweep_count": 0,
                "keywords_display": ", ".join(keywords_display) if keywords_display else "",
                "channel": "",
                "stim_channel": "",
                "events_channel": "",
                "experiment_name": "",
                "strain": "",
                "stim_type": "",
                "power": path_info["power_levels"][0] if path_info["power_levels"] else "",
                "sex": "",
                "animal_id": path_info["animal_ids"][0] if path_info["animal_ids"] else "",
                "status": "pending",
            }
            new_entries.append(entry)
            existing_paths.add(normalized)
            current += 1

            if progress_callback and current % 20 == 0:
                progress_callback(current, total, f"Discovering files... {current}/{total}")

        # Photometry files
        for fp_path in photometry_files:
            normalized = str(fp_path.resolve())
            if normalized in existing_paths:
                current += 1
                continue

            fp_info = project_builder.extract_photometry_info(fp_path)
            if fp_info:
                path_info = extract_path_keywords(fp_path, root)
                keywords_display = []
                if path_info.get("relative_path"):
                    keywords_display.append(path_info["relative_path"])

                regions_str = ", ".join(fp_info.get("signal_columns", [])) or "Unknown"
                channel_info = f"{fp_info.get('region_count', 0)} regions ({regions_str})"

                entry = {
                    "file_path": str(fp_path),
                    "file_name": fp_info.get("file_name", fp_path.name),
                    "file_type": "photometry",
                    "protocol": fp_info.get("protocol", "Neurophotometrics"),
                    "channel_count": fp_info.get("region_count", 0),
                    "sweep_count": 1,
                    "channel": channel_info,
                    "stim_channel": "",
                    "events_channel": "",
                    "experiment_name": "",
                    "strain": "",
                    "stim_type": "",
                    "power": path_info["power_levels"][0] if path_info["power_levels"] else "",
                    "sex": "",
                    "animal_id": path_info["animal_ids"][0] if path_info["animal_ids"] else "",
                    "status": "pending",
                    "keywords_display": ", ".join(keywords_display) if keywords_display else "",
                }
                new_entries.append(entry)
                existing_paths.add(normalized)

            current += 1
            if progress_callback:
                progress_callback(current, total, f"Discovering files... {current}/{total}")

        if progress_callback:
            progress_callback(total, total, f"Found {len(new_entries)} new files")

        return new_entries

    def load_file_metadata(
        self,
        file_entries: Optional[List[Dict]] = None,
        progress_callback=None,
    ) -> int:
        """Load detailed metadata (protocol, channels, sweeps) for file entries."""
        from core.fast_abf_reader import read_file_metadata_fast

        targets = file_entries if file_entries is not None else self._files
        updated = 0
        total = len(targets)

        for i, entry in enumerate(targets):
            file_path = Path(entry.get("file_path", ""))
            if not file_path.exists():
                continue
            if entry.get("file_type") == "photometry":
                continue
            if entry.get("protocol") and entry["protocol"] not in ("", "Loading..."):
                continue

            metadata = read_file_metadata_fast(file_path)
            if metadata:
                entry["protocol"] = metadata.get("protocol", "Unknown")
                entry["channel_count"] = metadata.get("channel_count", 0)
                entry["sweep_count"] = metadata.get("sweep_count", 0)
                if metadata.get("stim_frequency") and not entry.get("stim_type"):
                    entry["stim_type"] = metadata["stim_frequency"]
                if metadata.get("stim_channels") and not entry.get("events_channel"):
                    entry["events_channel"] = ", ".join(metadata["stim_channels"])
                updated += 1
                self._dirty = True

                # Update DB
                eid = self._resolve_experiment_id(str(file_path))
                if eid:
                    self.store.update_experiment(eid, {
                        "protocol": entry["protocol"],
                        "channel_count": entry["channel_count"],
                        "sweep_count": entry["sweep_count"],
                        "stim_type": entry.get("stim_type", ""),
                        "events_channel": entry.get("events_channel", ""),
                    })

            if progress_callback and (i % 10 == 0 or i == total - 1):
                progress_callback(i + 1, total, f"Loading metadata... {i + 1}/{total}")

        return updated

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def get_project_files(self) -> List[Dict[str, Any]]:
        """Return the current list of experiments with all metadata."""
        return [dict(f) for f in self._files]

    def get_file_by_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get experiment(s) by file path. Returns first match."""
        file_path_resolved = str(Path(file_path).resolve())
        for f in self._files:
            if str(Path(f.get("file_path", "")).resolve()) == file_path_resolved:
                return dict(f)
        return None

    def update_file_metadata(
        self, file_path: str, updates: Dict[str, Any],
        provenance: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update metadata for an experiment.

        Accepts ANY field — standard fields update columns,
        unknown fields become dynamic columns via ALTER TABLE.
        """
        eid = self._resolve_experiment_id(file_path)
        if eid is None:
            return None

        self.store.update_experiment(eid, updates)

        # Record source links if provenance provided
        if provenance:
            animal_id = ''
            exp = self.store.get_experiment(eid)
            if exp:
                animal_id = exp.get('animal_id', '')

            source_path = provenance.get('source_detail', '')
            if source_path and animal_id:
                source_id = self.store.add_source(source_path)
                for key, value in updates.items():
                    self.store.add_link(
                        source_id=source_id,
                        animal_id=animal_id,
                        field=key,
                        value=str(value),
                        experiment_id=eid,
                        confidence=float(provenance.get("confidence", 0.8)),
                    )

        # Update in-memory list
        file_path_resolved = str(Path(self._to_absolute(file_path)).resolve())
        for f in self._files:
            if str(Path(f.get("file_path", "")).resolve()) == file_path_resolved:
                for key, value in updates.items():
                    f[key] = value
                self._dirty = True
                return dict(f)

        # Not in memory — refresh
        self._refresh_files_from_db()
        self._dirty = True
        return self.get_file_by_path(file_path)

    def batch_update(self, file_paths: List[str], updates: Dict[str, Any],
                     provenance: Optional[Dict[str, str]] = None) -> int:
        """Apply the same metadata updates to multiple files."""
        count = 0
        for fp in file_paths:
            result = self.update_file_metadata(fp, updates, provenance=provenance)
            if result is not None:
                count += 1
        return count

    def add_files(self, file_entries: List[Dict[str, Any]]) -> int:
        """Add new experiment entries to the DB (and in-memory list)."""
        added = 0
        for entry in file_entries:
            fp = entry.get("file_path", "")
            if not fp:
                continue

            # Check if already exists
            existing = self.store.get_experiment_by_key(
                fp, entry.get('channel', ''), entry.get('animal_id', ''),
            )
            if existing:
                continue

            self.store.upsert_experiment(entry)
            added += 1
            self._dirty = True

        if added > 0:
            self._refresh_files_from_db()
        return added

    def remove_file(self, file_path: str) -> bool:
        """Remove experiment(s) for a file path."""
        exps = self.store.get_experiments_by_file(file_path)
        if not exps:
            abs_path = self._to_absolute(file_path)
            exps = self.store.get_experiments_by_file(abs_path)

        if not exps:
            return False

        for exp in exps:
            self.store.delete_experiment(exp['experiment_id'])

        self._refresh_files_from_db()
        self._dirty = True
        return True

    # ------------------------------------------------------------------
    # Source documents
    # ------------------------------------------------------------------

    def add_source(self, path: str) -> int:
        """Register a reference document. Returns source_id."""
        file_hash = ''
        try:
            file_hash = self.file_hash(Path(path))
        except OSError:
            pass
        return self.store.add_source(path, file_hash)

    def link_source(
        self, source_id: int, animal_id: str, field: str, value: str,
        experiment_id: Optional[int] = None,
        location: str = '', confidence: float = 0.5,
    ) -> int:
        """Create a source_link. Returns link_id."""
        return self.store.add_link(
            source_id=source_id, animal_id=animal_id, field=field,
            value=value, experiment_id=experiment_id,
            location=location, confidence=confidence,
        )

    def get_source_links(
        self, animal_id: Optional[str] = None,
        experiment_id: Optional[int] = None,
        field: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query source links."""
        return self.store.get_links(
            animal_id=animal_id, experiment_id=experiment_id, field=field,
        )

    def get_sources(self) -> List[Dict[str, Any]]:
        """List registered source documents."""
        return self.store.get_sources()

    # ------------------------------------------------------------------
    # Cross-experiment queries
    # ------------------------------------------------------------------

    def get_experiments_for_animal(self, animal_id: str) -> List[Dict[str, Any]]:
        """Get all experiments for an animal across all files."""
        return self.store.get_experiments_for_animal(animal_id)

    # ------------------------------------------------------------------
    # Custom columns
    # ------------------------------------------------------------------

    def add_custom_column(
        self, column_key: str, display_name: str,
        column_type: str = 'text', sort_order: int = 0,
    ) -> bool:
        """Add a custom column via ALTER TABLE."""
        sql_type = 'TEXT'
        if column_type == 'number':
            sql_type = 'REAL'
        elif column_type == 'boolean':
            sql_type = 'INTEGER'
        return self.store.add_column(column_key, sql_type)

    def get_custom_columns(self) -> List[Dict[str, Any]]:
        """Get dynamic (user-added) column definitions."""
        return self.store.get_dynamic_columns()

    # ------------------------------------------------------------------
    # Provenance (via source_links)
    # ------------------------------------------------------------------

    def get_provenance(
        self, file_path: str, field: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get source links for an experiment (legacy provenance API)."""
        eid = self._resolve_experiment_id(file_path)
        if eid is None:
            return []
        exp = self.store.get_experiment(eid)
        if not exp:
            return []

        animal_id = exp.get('animal_id', '')
        if animal_id:
            return self.store.get_links(
                animal_id=animal_id, experiment_id=eid, field=field,
            )
        return self.store.get_links(experiment_id=eid, field=field)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def get_metadata_completeness(self) -> Dict[str, Any]:
        """Get percentage filled for each metadata column."""
        return self.store.get_metadata_completeness()

    def get_unique_values(self, field: str) -> List[str]:
        """Get sorted unique non-empty values for a metadata field."""
        return self.store.get_unique_values(field)

    def get_file_count(self) -> int:
        """Get total number of experiments."""
        return self.store.get_experiment_count()

    # ------------------------------------------------------------------
    # Grouped view
    # ------------------------------------------------------------------

    def get_files_grouped(self) -> Dict[str, Any]:
        """Return experiments grouped by file_path with summary stats."""
        return self.store.get_experiments_grouped()

    # ------------------------------------------------------------------
    # Backup
    # ------------------------------------------------------------------

    def backup(self, trigger_event: str = "manual") -> str:
        return self.store.backup(trigger_event)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def file_hash(path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
