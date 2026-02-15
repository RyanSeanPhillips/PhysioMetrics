"""
Project service — pure Python, no PyQt6 imports.

Provides high-level operations for scanning folders, parsing notes,
reading/writing project metadata, and querying file metadata.
Designed to be callable from MCP (no Qt dependency) and from the
ProjectViewModel (Qt bridge).
"""

import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime

from core.project_manager import ProjectManager


# Metadata column keys for the master file list
METADATA_COLUMNS = [
    'file_name', 'file_path', 'experiment', 'strain', 'stim_type',
    'power', 'sex', 'animal_id', 'protocol', 'channel', 'stim_channel',
    'events_channel', 'channel_count', 'sweep_count', 'keywords',
    'status', 'exports', 'linked_notes',
]

# Editable metadata columns (subset that Claude / user can write to)
EDITABLE_COLUMNS = [
    'experiment', 'strain', 'stim_type', 'power', 'sex', 'animal_id',
    'channel', 'stim_channel', 'events_channel', 'status', 'tags',
    'notes', 'group', 'weight', 'age', 'date_recorded',
]


class ProjectService:
    """
    Pure-Python service for project metadata operations.

    Wraps ProjectManager (save/load) and adds:
    - Folder scanning with metadata extraction
    - Notes file parsing (Excel/CSV/TXT/DOCX)
    - Fuzzy matching of notes to recording files
    - CRUD operations on file metadata
    - Completeness and uniqueness queries
    """

    def __init__(self, project_manager: Optional[ProjectManager] = None):
        self._pm = project_manager or ProjectManager()
        self._files: List[Dict[str, Any]] = []
        self._project_path: Optional[Path] = None
        self._project_name: str = ""
        self._data_directory: Optional[Path] = None
        self._experiments: List[Dict] = []
        self._notes_files_data: List[Dict] = []
        self._dirty = False
        self._cache = None  # Lazy-loaded ProjectCacheSQLite

    # ------------------------------------------------------------------
    # Project open / save / load
    # ------------------------------------------------------------------

    def open_project(self, folder: Path) -> Dict[str, Any]:
        """
        Open or create a project for a folder.

        If a .physiometrics file exists in the folder, load it.
        Otherwise, create a new empty project.

        Returns:
            Summary dict with project_name, file_count, data_directory.
        """
        folder = Path(folder)
        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder}")

        # Look for existing .physiometrics file
        pm_files = list(folder.glob("*.physiometrics"))

        if pm_files:
            # Load the most recently modified one
            pm_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return self.load_project(pm_files[0])

        # No project file — create new empty project
        self._project_name = folder.name
        self._data_directory = folder
        self._project_path = None
        self._files = []
        self._experiments = []
        self._notes_files_data = []
        self._dirty = False

        return {
            "project_name": self._project_name,
            "data_directory": str(folder),
            "file_count": 0,
            "status": "created_new",
        }

    def load_project(self, project_path: Path) -> Dict[str, Any]:
        """Load a project from a .physiometrics file."""
        project_path = Path(project_path)
        data = self._pm.load_project(project_path)

        self._project_path = project_path
        self._project_name = data.get("project_name", project_path.stem)
        self._data_directory = Path(data.get("data_directory", project_path.parent))
        self._files = data.get("files", [])
        self._experiments = data.get("experiments", [])
        self._notes_files_data = data.get("notes_files", [])
        self._dirty = False

        # Ensure file paths are strings for consistency
        for f in self._files:
            if "file_path" in f and not isinstance(f["file_path"], str):
                f["file_path"] = str(f["file_path"])

        return {
            "project_name": self._project_name,
            "data_directory": str(self._data_directory),
            "file_count": len(self._files),
            "experiment_count": len(self._experiments),
            "status": "loaded",
        }

    def save_project(self, name: Optional[str] = None) -> str:
        """Save the current project to disk."""
        if self._data_directory is None:
            raise RuntimeError("No project open — call open_project() first")

        project_name = name or self._project_name or self._data_directory.name
        self._project_name = project_name

        self._project_path = self._pm.save_project(
            project_name=project_name,
            data_directory=self._data_directory,
            files_data=self._files,
            experiments=self._experiments,
            notes_files=self._notes_files_data,
        )
        self._dirty = False
        return str(self._project_path)

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @property
    def project_path(self) -> Optional[Path]:
        return self._project_path

    @property
    def data_directory(self) -> Optional[Path]:
        return self._data_directory

    # ------------------------------------------------------------------
    # Relative / absolute path helpers
    # ------------------------------------------------------------------

    def _to_relative(self, abs_path: str) -> str:
        """Convert an absolute path to a relative path from the project data directory.

        Returns the original path if it can't be made relative (different drive, etc.).
        """
        if not self._data_directory or not abs_path:
            return abs_path
        try:
            rel = Path(abs_path).relative_to(self._data_directory)
            return str(rel).replace("\\", "/")
        except (ValueError, TypeError):
            return abs_path

    def _to_absolute(self, rel_or_abs: str) -> str:
        """Resolve a relative-or-absolute path against the project data directory.

        If the path is already absolute (or data_directory is unset), returns as-is.
        """
        if not rel_or_abs:
            return rel_or_abs
        p = Path(rel_or_abs)
        if p.is_absolute():
            return str(p)
        if self._data_directory:
            return str(self._data_directory / p)
        return rel_or_abs

    # ------------------------------------------------------------------
    # Cache (lazy-loaded)
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_dir_for_project(data_directory: Path) -> Path:
        """Get the AppData cache directory for a project, keyed by path hash."""
        import os
        appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        folder_hash = hashlib.sha256(str(data_directory.resolve()).encode()).hexdigest()[:16]
        return appdata / "PhysioMetrics" / "project_cache" / folder_hash

    @property
    def cache(self):
        """Lazy-load the project cache DB. Stored in %APPDATA%/PhysioMetrics/project_cache/<hash>/."""
        if self._cache is None and self._data_directory is not None:
            from core.adapters.project_cache_sqlite import ProjectCacheSQLite
            cache_dir = self._cache_dir_for_project(self._data_directory)
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = cache_dir / "cache.db"
            self._cache = ProjectCacheSQLite(db_path)
            # Store a breadcrumb so we can map hash back to project
            self._cache.set_knowledge("_project_path", str(self._data_directory.resolve()))
        return self._cache

    def get_cached_notes(self, path: Path) -> Optional[List[Dict[str, Any]]]:
        """Check cache for parsed notes. Returns cached content if hash matches, else None."""
        if self.cache is None:
            return None
        current_hash = self.file_hash(path)
        cached = self.cache.get_notes_cache(current_hash)
        if cached:
            return cached["content"]
        return None

    def cache_notes_result(self, path: Path, content: List[Dict], abf_refs: List[str]) -> None:
        """Store parsed notes in the cache."""
        if self.cache is None:
            return
        current_hash = self.file_hash(path)
        self.cache.save_notes_cache(str(path), current_hash, content, abf_refs)

    def save_extraction_pattern(self, type: str, source: str, target: str,
                                 extractor: str, confidence: float = 0.5,
                                 notes: Optional[str] = None) -> Optional[int]:
        """Store an extraction pattern. Returns pattern_id or None."""
        if self.cache is None:
            return None
        return self.cache.save_pattern(type, source, target, extractor, confidence, notes)

    def get_extraction_patterns(self, target_field: Optional[str] = None,
                                 type: Optional[str] = None,
                                 min_confidence: float = 0.0) -> List[Dict]:
        """Retrieve cached extraction patterns."""
        if self.cache is None:
            return []
        return self.cache.get_patterns(target_field, type, min_confidence)

    def add_vocabulary_value(self, field: str, value: str) -> None:
        """Record a known value for a metadata field."""
        if self.cache is not None:
            self.cache.add_vocabulary(field, value)

    def get_vocabulary(self, field: Optional[str] = None) -> List[Dict]:
        """Get known vocabulary values."""
        if self.cache is None:
            return []
        return self.cache.get_vocabulary(field)

    def set_knowledge(self, key: str, value: Any) -> None:
        """Store a key-value pair in the cache."""
        if self.cache is not None:
            self.cache.set_knowledge(key, value)

    def get_knowledge(self, key: str) -> Optional[Any]:
        """Retrieve a cached key-value pair."""
        if self.cache is None:
            return None
        return self.cache.get_knowledge(key)

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
        """
        Discover recording files in a folder tree.

        Args:
            root: Folder to scan (defaults to project data_directory).
            recursive: Whether to search subdirectories.
            file_types: List of types to scan for ('abf', 'smrx', 'edf', 'photometry').
            progress_callback: Optional callback(current, total, message).

        Returns:
            List of newly discovered file dicts (not yet added to project).
        """
        from core import project_builder
        from core.fast_abf_reader import extract_path_keywords

        root = Path(root) if root else self._data_directory
        if root is None:
            raise RuntimeError("No folder to scan — pass root or open a project first")

        files = project_builder.discover_files(
            str(root), recursive=recursive, file_types=file_types
        )

        abf_files = files.get("abf_files", [])
        smrx_files = files.get("smrx_files", [])
        edf_files = files.get("edf_files", [])
        photometry_files = files.get("photometry_files", [])

        all_data_files = abf_files + smrx_files + edf_files

        # Build set of existing paths
        existing_paths = {
            str(Path(f["file_path"]).resolve())
            for f in self._files
            if f.get("file_path")
        }

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
                "stim_channels": [],
                "stim_frequency": "",
                "path_keywords": path_info,
                "keywords_display": ", ".join(keywords_display) if keywords_display else "",
                "channel": "",
                "stim_channel": "",
                "events_channel": "",
                "experiment": "",
                "strain": "",
                "stim_type": "",
                "power": path_info["power_levels"][0] if path_info["power_levels"] else "",
                "sex": "",
                "animal_id": path_info["animal_ids"][0] if path_info["animal_ids"] else "",
                "status": "pending",
                "exports": {},
                "linked_notes": "",
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
                    "experiment": "",
                    "strain": "",
                    "stim_type": "",
                    "power": path_info["power_levels"][0] if path_info["power_levels"] else "",
                    "sex": "",
                    "animal_id": path_info["animal_ids"][0] if path_info["animal_ids"] else "",
                    "status": "pending",
                    "exports": {},
                    "linked_notes": "",
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
        """
        Load detailed metadata (protocol, channels, sweeps) for file entries.

        Operates on self._files if file_entries is None.

        Returns:
            Number of files updated.
        """
        from core.fast_abf_reader import read_file_metadata_fast

        targets = file_entries if file_entries is not None else self._files
        updated = 0
        total = len(targets)

        for i, entry in enumerate(targets):
            file_path = Path(entry.get("file_path", ""))
            if not file_path.exists():
                continue
            if entry.get("file_type") == "photometry":
                continue  # Already has metadata from scan
            if entry.get("protocol") and entry["protocol"] not in ("", "Loading..."):
                continue  # Already loaded

            metadata = read_file_metadata_fast(file_path)
            if metadata:
                entry["protocol"] = metadata.get("protocol", "Unknown")
                entry["channel_count"] = metadata.get("channel_count", 0)
                entry["sweep_count"] = metadata.get("sweep_count", 0)
                entry["stim_channels"] = metadata.get("stim_channels", [])
                entry["stim_frequency"] = metadata.get("stim_frequency", "")
                if metadata.get("stim_frequency") and not entry.get("stim_type"):
                    entry["stim_type"] = metadata["stim_frequency"]
                if metadata.get("stim_channels") and not entry.get("events_channel"):
                    entry["events_channel"] = ", ".join(metadata["stim_channels"])
                updated += 1
                self._dirty = True

            if progress_callback and (i % 10 == 0 or i == total - 1):
                progress_callback(i + 1, total, f"Loading metadata... {i + 1}/{total}")

        return updated

    # ------------------------------------------------------------------
    # Notes file reading and matching
    # ------------------------------------------------------------------

    def read_notes_file(self, path: Path) -> List[Dict[str, Any]]:
        """
        Read and parse a notes file (Excel/CSV/TXT/DOCX).

        Checks the cache first — if the file hash matches a cached parse,
        returns the cached result immediately. Otherwise parses and caches.

        Returns a list of dicts, each representing a row or text block
        with extracted content and ABF file references found.
        """
        path = Path(path)

        # Check cache first
        cached = self.get_cached_notes(path)
        if cached is not None:
            return cached

        suffix = path.suffix.lower()
        entries = []

        try:
            if suffix in (".xlsx", ".xls"):
                entries = self._parse_excel_notes(path)
            elif suffix == ".csv":
                entries = self._parse_csv_notes(path)
            elif suffix == ".txt":
                entries = self._parse_text_notes(path)
            elif suffix == ".docx":
                entries = self._parse_docx_notes(path)
        except Exception as e:
            print(f"[project-service] Error reading notes file {path}: {e}")
            return [{"error": str(e), "path": str(path)}]

        # Cache the result
        all_refs = []
        for entry in entries:
            all_refs.extend(entry.get("abf_references", []))
        self.cache_notes_result(path, entries, sorted(set(all_refs)))

        return entries

    def _parse_excel_notes(self, path: Path) -> List[Dict]:
        """Parse Excel file, returning rows as dicts with ABF refs detected."""
        import pandas as pd

        entries = []
        sheets = pd.read_excel(path, sheet_name=None, nrows=1000)

        for sheet_name, df in sheets.items():
            # Convert all cells to strings and search for ABF references
            abf_refs = set()
            for col in df.columns:
                for val in df[col].astype(str):
                    refs = self._extract_abf_references(val)
                    abf_refs.update(refs)

            entries.append({
                "source_file": str(path),
                "source_name": path.name,
                "sheet_name": sheet_name,
                "row_count": len(df),
                "columns": list(df.columns),
                "abf_references": sorted(abf_refs),
                "content_preview": df.head(5).to_dict(orient="records"),
            })

        return entries

    def _parse_csv_notes(self, path: Path) -> List[Dict]:
        """Parse CSV file."""
        import pandas as pd

        df = pd.read_csv(path, nrows=1000, encoding="utf-8", on_bad_lines="skip")

        abf_refs = set()
        for col in df.columns:
            for val in df[col].astype(str):
                refs = self._extract_abf_references(val)
                abf_refs.update(refs)

        return [{
            "source_file": str(path),
            "source_name": path.name,
            "row_count": len(df),
            "columns": list(df.columns),
            "abf_references": sorted(abf_refs),
            "content_preview": df.head(5).to_dict(orient="records"),
        }]

    def _parse_text_notes(self, path: Path) -> List[Dict]:
        """Parse plain text file."""
        content = path.read_text(encoding="utf-8", errors="replace")
        abf_refs = self._extract_abf_references(content)

        return [{
            "source_file": str(path),
            "source_name": path.name,
            "content": content[:5000],  # Limit for MCP response size
            "line_count": content.count("\n") + 1,
            "abf_references": sorted(abf_refs),
        }]

    def _parse_docx_notes(self, path: Path) -> List[Dict]:
        """Parse Word document."""
        try:
            from docx import Document
        except ImportError:
            return [{"error": "python-docx not installed", "path": str(path)}]

        doc = Document(str(path))

        content_parts = []
        for p in doc.paragraphs:
            if p.text.strip():
                content_parts.append(p.text)

        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                content_parts.append(row_text)

        content = "\n".join(content_parts)
        abf_refs = self._extract_abf_references(content)

        return [{
            "source_file": str(path),
            "source_name": path.name,
            "content": content[:5000],
            "paragraph_count": len(doc.paragraphs),
            "table_count": len(doc.tables),
            "abf_references": sorted(abf_refs),
        }]

    @staticmethod
    def _extract_abf_references(text: str) -> Set[str]:
        """
        Extract ABF file stem references from text.

        Looks for patterns like:
        - "25708007" (8-digit numbers)
        - "25708007.abf"
        - Numeric sequences that look like ABF file IDs
        """
        refs = set()

        # Match .abf file references
        abf_matches = re.findall(r"(\w+)\.abf", text, re.IGNORECASE)
        refs.update(abf_matches)

        # Match 6-10 digit numbers (common ABF file naming)
        num_matches = re.findall(r"\b(\d{6,10})\b", text)
        refs.update(num_matches)

        return refs

    def match_notes_to_files(
        self,
        notes_entries: List[Dict],
        fuzzy_range: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Match notes entries to recording files using multiple strategies:

        1. Exact stem match: "2024_03_21_0017" matches file stem exactly
        2. Suffix match: "017" or "17" matches the last N digits of a file stem
        3. Fuzzy numeric: look for numeric neighbors within +-fuzzy_range

        Returns:
            List of match dicts with file_path, file_name, match_type, matched_ref, notes_source.
        """
        matches = []

        # Build lookups
        stem_to_files: Dict[str, List[Dict]] = {}
        suffix_to_files: Dict[str, List[Dict]] = {}  # last 3-5 digits → files

        for f in self._files:
            stem = Path(f.get("file_name", "")).stem
            if stem:
                stem_to_files.setdefault(stem, []).append(f)
                # Extract trailing digits for suffix matching
                trailing = re.search(r"(\d{3,5})$", stem)
                if trailing:
                    suffix = trailing.group(1)
                    suffix_to_files.setdefault(suffix, []).append(f)
                    # Also store shorter suffixes (last 3 and last 4)
                    if len(suffix) >= 4:
                        suffix_to_files.setdefault(suffix[-3:], []).append(f)
                    if len(suffix) >= 5:
                        suffix_to_files.setdefault(suffix[-4:], []).append(f)

        seen_matches = set()  # (file_path, ref) to avoid duplicates

        for note in notes_entries:
            abf_refs = note.get("abf_references", [])
            source = note.get("source_name", "unknown")

            for ref in abf_refs:
                # Strategy 1: Exact stem match
                if ref in stem_to_files:
                    for f in stem_to_files[ref]:
                        key = (f.get("file_path", ""), ref)
                        if key not in seen_matches:
                            seen_matches.add(key)
                            matches.append({
                                "file_path": f.get("file_path", ""),
                                "file_name": f.get("file_name", ""),
                                "match_type": "exact",
                                "matched_ref": ref,
                                "notes_source": source,
                            })
                    continue

                # Strategy 2: Suffix match (last N digits)
                if ref.isdigit() and len(ref) <= 5 and ref in suffix_to_files:
                    for f in suffix_to_files[ref]:
                        key = (f.get("file_path", ""), ref)
                        if key not in seen_matches:
                            seen_matches.add(key)
                            matches.append({
                                "file_path": f.get("file_path", ""),
                                "file_name": f.get("file_name", ""),
                                "match_type": "suffix",
                                "matched_ref": ref,
                                "notes_source": source,
                            })
                    continue

                # Strategy 3: Fuzzy numeric match
                numeric_parts = re.findall(r"\d+", ref)
                if not numeric_parts:
                    continue

                main_numeric = max(numeric_parts, key=len)
                numeric_idx = ref.find(main_numeric)

                try:
                    base_num = int(main_numeric)
                except ValueError:
                    continue

                for delta in range(-fuzzy_range, fuzzy_range + 1):
                    if delta == 0:
                        continue
                    candidate_num = base_num + delta
                    if candidate_num < 0:
                        continue
                    candidate_str = str(candidate_num).zfill(len(main_numeric))
                    candidate_stem = ref[:numeric_idx] + candidate_str + ref[numeric_idx + len(main_numeric):]

                    if candidate_stem in stem_to_files:
                        for f in stem_to_files[candidate_stem]:
                            key = (f.get("file_path", ""), candidate_stem)
                            if key not in seen_matches:
                                seen_matches.add(key)
                                matches.append({
                                    "file_path": f.get("file_path", ""),
                                    "file_name": f.get("file_name", ""),
                                    "match_type": "fuzzy",
                                    "matched_ref": candidate_stem,
                                    "original_ref": ref,
                                    "notes_source": source,
                                })

        return matches

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def get_project_files(self) -> List[Dict[str, Any]]:
        """Return the current list of files with all metadata."""
        return [dict(f) for f in self._files]

    def get_file_by_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get a single file entry by its path."""
        file_path_resolved = str(Path(file_path).resolve())
        for f in self._files:
            if str(Path(f.get("file_path", "")).resolve()) == file_path_resolved:
                return dict(f)
        return None

    def update_file_metadata(
        self, file_path: str, updates: Dict[str, Any],
        provenance: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update metadata for a single file.

        Args:
            file_path: Path to the file to update.
            updates: Dict of field->value pairs to update.
            provenance: Optional provenance info: {source_type, source_detail, reason}.

        Returns:
            Updated file dict, or None if file not found.
        """
        # Try both absolute resolution and relative-path resolution
        file_path_resolved = str(Path(self._to_absolute(file_path)).resolve())

        for f in self._files:
            if str(Path(f.get("file_path", "")).resolve()) == file_path_resolved:
                rel_path = self._to_relative(f.get("file_path", ""))
                # Only update editable fields
                for key, value in updates.items():
                    if key in EDITABLE_COLUMNS or key.startswith("custom_"):
                        f[key] = value
                        # Record provenance if available
                        if provenance:
                            self._record_provenance(
                                file_path=rel_path,
                                field=key,
                                value=str(value),
                                source_type=provenance.get("source_type", "user"),
                                source_detail=provenance.get("source_detail", ""),
                                source_preview=provenance.get("source_preview", ""),
                                confidence=float(provenance.get("confidence", 0.8)),
                                reason=provenance.get("reason", ""),
                            )
                self._dirty = True
                return dict(f)

        return None

    def batch_update(self, file_paths: List[str], updates: Dict[str, Any]) -> int:
        """
        Apply the same metadata updates to multiple files.

        Returns:
            Number of files updated.
        """
        count = 0
        for fp in file_paths:
            result = self.update_file_metadata(fp, updates)
            if result is not None:
                count += 1
        return count

    def add_files(self, file_entries: List[Dict[str, Any]]) -> int:
        """
        Add new file entries to the project.

        Returns:
            Number of files added.
        """
        existing_paths = {
            str(Path(f["file_path"]).resolve())
            for f in self._files
            if f.get("file_path")
        }

        added = 0
        for entry in file_entries:
            fp = entry.get("file_path", "")
            if fp and str(Path(fp).resolve()) not in existing_paths:
                self._files.append(dict(entry))
                existing_paths.add(str(Path(fp).resolve()))
                added += 1
                self._dirty = True

        return added

    def remove_file(self, file_path: str) -> bool:
        """Remove a file from the project by path."""
        file_path_resolved = str(Path(file_path).resolve())
        for i, f in enumerate(self._files):
            if str(Path(f.get("file_path", "")).resolve()) == file_path_resolved:
                del self._files[i]
                self._dirty = True
                return True
        return False

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def get_metadata_completeness(self) -> Dict[str, Any]:
        """
        Get percentage filled for each metadata column.

        Returns:
            Dict with keys: per_column (dict of field->pct), overall_pct, total_files.
        """
        if not self._files:
            return {"per_column": {}, "overall_pct": 0.0, "total_files": 0}

        fields = ["experiment", "strain", "stim_type", "power", "sex",
                   "animal_id", "channel", "stim_channel", "protocol"]

        per_column = {}
        total_filled = 0
        total_cells = 0

        for field in fields:
            filled = sum(
                1 for f in self._files
                if f.get(field) and str(f[field]).strip()
                and str(f[field]) not in ("Loading...", "Unknown", "pending")
            )
            pct = (filled / len(self._files)) * 100 if self._files else 0
            per_column[field] = {
                "filled": filled,
                "total": len(self._files),
                "percent": round(pct, 1),
            }
            total_filled += filled
            total_cells += len(self._files)

        overall = (total_filled / total_cells * 100) if total_cells > 0 else 0

        return {
            "per_column": per_column,
            "overall_pct": round(overall, 1),
            "total_files": len(self._files),
        }

    def get_unique_values(self, field: str) -> List[str]:
        """
        Get sorted unique non-empty values for a metadata field.

        Useful for autocomplete, validation, and pattern detection.
        """
        values = set()
        for f in self._files:
            val = f.get(field, "")
            if val and str(val).strip() and str(val) not in ("Loading...", "Unknown"):
                values.add(str(val).strip())
        return sorted(values)

    def get_file_count(self) -> int:
        """Get total number of files in the project."""
        return len(self._files)

    # ------------------------------------------------------------------
    # Grouped file listing
    # ------------------------------------------------------------------

    def get_files_grouped(self) -> Dict[str, Any]:
        """
        Return files grouped by folder with summary stats.

        Much more context-efficient than a flat file list — saves ~80 chars/file.
        """
        if not self._files:
            return {
                "project_root": str(self._data_directory) if self._data_directory else "",
                "total_files": 0,
                "folders": [],
            }

        folder_map: Dict[str, List[Dict]] = {}
        for f in self._files:
            fp = f.get("file_path", "")
            rel = self._to_relative(fp)
            parent = str(Path(rel).parent).replace("\\", "/")
            if parent == ".":
                parent = ""
            folder_map.setdefault(parent, []).append(f)

        fields_to_check = ["experiment", "strain", "stim_type", "power", "sex",
                           "animal_id", "channel", "protocol"]

        folders = []
        for folder_path in sorted(folder_map.keys()):
            files_in_folder = folder_map[folder_path]
            count = len(files_in_folder)

            # Compute fill percentage
            total_cells = count * len(fields_to_check)
            filled_cells = 0
            for f in files_in_folder:
                for field in fields_to_check:
                    val = f.get(field, "")
                    if val and str(val).strip() and str(val) not in ("Loading...", "Unknown", "pending"):
                        filled_cells += 1
            filled_pct = round((filled_cells / total_cells * 100) if total_cells else 0, 1)

            # Collect common values (show if all files share them)
            common = {}
            for field in ["strain", "animal_id", "stim_type", "power", "experiment"]:
                vals = set()
                for f in files_in_folder:
                    v = f.get(field, "")
                    if v and str(v).strip():
                        vals.add(str(v).strip())
                if len(vals) == 1:
                    common[field] = vals.pop()

            entry = {
                "path": folder_path,
                "count": count,
                "filled_pct": filled_pct,
            }
            entry.update(common)
            # List file names (compact)
            entry["files"] = [f.get("file_name", "") for f in files_in_folder]
            folders.append(entry)

        return {
            "project_root": str(self._data_directory) if self._data_directory else "",
            "total_files": len(self._files),
            "folders": folders,
        }

    # ------------------------------------------------------------------
    # Notes file discovery (with scoring heuristics)
    # ------------------------------------------------------------------

    # Negative keywords in filenames — less likely to be experiment notes
    _NEGATIVE_KEYWORDS = {
        "consolidated", "export", "analysis", "output", "results", "summary",
        "backup", "copy", "old", "archive", "temp", "template",
    }

    def discover_notes_files(self, root: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Discover notes files (Excel, CSV, TXT, DOCX) in the project folder.

        Returns candidates with confidence scores and classification reasons.
        """
        from core import project_builder

        root = Path(root) if root else self._data_directory
        if root is None:
            return []

        files = project_builder.discover_files(str(root), recursive=True, file_types=["notes"])
        notes_files = files.get("notes_files", [])

        # Build data file folder set for proximity scoring
        data_folders: Set[Path] = set()
        for f in self._files:
            fp = f.get("file_path", "")
            if fp:
                data_folders.add(Path(fp).parent.resolve())

        result = []
        for nf in notes_files:
            try:
                stat = nf.stat()
                size_bytes = stat.st_size
                size_kb = size_bytes / 1024
                modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            except OSError:
                size_bytes = 0
                size_kb = 0
                modified = ""

            # --- Scoring ---
            confidence = 0.5
            reasons = []

            # Size heuristic
            if size_kb < 100:
                confidence += 0.15
                reasons.append("small file (<100KB)")
            elif size_kb < 500:
                confidence += 0.05
                reasons.append("medium file (100-500KB)")
            else:
                confidence -= 0.2
                reasons.append("large file (>500KB) — may be data export")

            # Proximity to data files
            nf_parent = nf.parent.resolve()
            min_hops = self._folder_distance(nf_parent, data_folders, root)
            if min_hops <= 1:
                confidence += 0.15
                reasons.append(f"near data files ({min_hops} hop{'s' if min_hops != 1 else ''})")
            elif min_hops >= 3:
                confidence -= 0.1
                reasons.append(f"far from data files ({min_hops} hops)")

            # Negative keywords
            name_lower = nf.stem.lower()
            neg_hits = [kw for kw in self._NEGATIVE_KEYWORDS if kw in name_lower]
            if neg_hits:
                confidence -= 0.2
                reasons.append(f"negative keywords: {', '.join(neg_hits)}")

            # Positive keywords
            positive = ["log", "notes", "protocol", "record", "experiment", "animal"]
            pos_hits = [kw for kw in positive if kw in name_lower]
            if pos_hits:
                confidence += 0.15
                reasons.append(f"positive keywords: {', '.join(pos_hits)}")

            confidence = max(0.0, min(1.0, confidence))

            entry = {
                "name": nf.name,
                "path": self._to_relative(str(nf)),
                "type": nf.suffix.lower().lstrip("."),
                "size": f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.1f} MB",
                "size_bytes": size_bytes,
                "modified": modified,
                "confidence": round(confidence, 2),
                "reasons": reasons,
            }

            # Content preview for Excel/CSV (first few rows)
            if nf.suffix.lower() in (".xlsx", ".xls", ".csv") and size_kb < 500:
                try:
                    preview = self._quick_preview(nf)
                    if preview:
                        entry["preview"] = preview
                except Exception:
                    pass

            result.append(entry)

        # Sort by confidence descending
        result.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return result

    @staticmethod
    def _folder_distance(target: Path, data_folders: Set[Path], root: Path) -> int:
        """Count minimum 'hops' from target folder to nearest data file folder."""
        if not data_folders:
            return 99
        try:
            target_parts = target.relative_to(root.resolve()).parts
        except ValueError:
            return 99

        min_dist = 99
        for df in data_folders:
            try:
                df_parts = df.relative_to(root.resolve()).parts
            except ValueError:
                continue
            # Count differing path components
            common = 0
            for a, b in zip(target_parts, df_parts):
                if a == b:
                    common += 1
                else:
                    break
            dist = (len(target_parts) - common) + (len(df_parts) - common)
            min_dist = min(min_dist, dist)
        return min_dist

    @staticmethod
    def _quick_preview(path: Path) -> Optional[Dict]:
        """Read first 5 rows of a tabular file for preview."""
        import pandas as pd
        suffix = path.suffix.lower()
        try:
            if suffix in (".xlsx", ".xls"):
                df = pd.read_excel(path, nrows=5)
            elif suffix == ".csv":
                df = pd.read_csv(path, nrows=5, encoding="utf-8", on_bad_lines="skip")
            else:
                return None
            return {
                "columns": list(df.columns),
                "row_count_preview": len(df),
                "sample_rows": df.head(3).to_dict(orient="records"),
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Pattern application (server-side, zero context cost)
    # ------------------------------------------------------------------

    def apply_patterns(self) -> Dict[str, Any]:
        """
        Apply all cached extraction patterns to unfilled metadata fields.

        Returns summary of what was applied (no per-file details to save context).
        """
        patterns = self.get_extraction_patterns(min_confidence=0.3)
        if not patterns:
            return {"applied": 0, "skipped": 0, "message": "No patterns cached"}

        applied = 0
        skipped = 0
        by_field: Dict[str, int] = {}

        for f in self._files:
            file_path = f.get("file_path", "")
            rel_path = self._to_relative(file_path)
            path_parts = Path(rel_path).parts

            for pattern in patterns:
                target = pattern["target"]
                # Skip if field already filled
                current = f.get(target, "")
                if current and str(current).strip() and str(current) not in ("", "Loading...", "Unknown"):
                    skipped += 1
                    continue

                value = self._try_apply_pattern(pattern, f, path_parts, rel_path)
                if value is not None:
                    f[target] = value
                    applied += 1
                    by_field[target] = by_field.get(target, 0) + 1
                    self._dirty = True

                    # Record provenance
                    self._record_provenance(
                        file_path=rel_path,
                        field=target,
                        value=str(value),
                        source_type="pattern",
                        source_detail=f"pattern#{pattern['pattern_id']}: {pattern['type']}:{pattern['source']}",
                        confidence=pattern.get("confidence", 0.5),
                        reason=f"Auto-applied pattern: {pattern.get('notes', pattern['extractor'])}",
                    )

                    # Increment pattern usage
                    if self.cache:
                        self.cache.increment_pattern_usage(pattern["pattern_id"])

        return {
            "applied": applied,
            "skipped": skipped,
            "by_field": by_field,
            "patterns_used": len(patterns),
        }

    def _try_apply_pattern(
        self, pattern: Dict, file_entry: Dict, path_parts: tuple, rel_path: str
    ) -> Optional[str]:
        """Try to apply a single pattern to a file entry. Returns extracted value or None."""
        ptype = pattern["type"]
        source = pattern["source"]
        extractor = pattern["extractor"]

        if ptype == "literal":
            # Literal value applies to all files matching source scope
            if source == "*" or source.lower() in rel_path.lower():
                return extractor.replace("literal:", "", 1) if extractor.startswith("literal:") else extractor

        elif ptype == "subfolder":
            # Check if source string appears in any path component
            for part in path_parts:
                if source.lower() == part.lower() or source.lower() in part.lower():
                    if extractor.startswith("literal:"):
                        return extractor[len("literal:"):]
                    elif extractor.startswith("regex:"):
                        match = re.search(extractor[len("regex:"):], part)
                        return match.group(0) if match else None
                    return part

        elif ptype == "filename":
            fname = file_entry.get("file_name", "")
            if extractor.startswith("regex:"):
                match = re.search(extractor[len("regex:"):], fname)
                return match.group(0) if match else None
            elif extractor.startswith("literal:"):
                return extractor[len("literal:"):]

        elif ptype == "regex":
            # Apply regex to full relative path
            match = re.search(source, rel_path)
            if match:
                if extractor.startswith("group:"):
                    try:
                        idx = int(extractor[len("group:"):])
                        return match.group(idx)
                    except (ValueError, IndexError):
                        return match.group(0)
                return match.group(0)

        return None

    # ------------------------------------------------------------------
    # Subrow support (multi-animal recordings)
    # ------------------------------------------------------------------

    def add_subrow(
        self, file_path: str, channel: str, animal_id: str = "",
        sex: str = "", group: str = "", **extra
    ) -> Optional[Dict[str, Any]]:
        """
        Create a child entry linked to a parent file for multi-animal recordings.

        Subrows inherit protocol/stim_type/power from parent. Stored as nested
        list in the file's metadata dict under 'subrows'.
        """
        file_path_resolved = str(Path(self._to_absolute(file_path)).resolve())

        for f in self._files:
            if str(Path(f.get("file_path", "")).resolve()) == file_path_resolved:
                subrows = f.setdefault("subrows", [])
                subrow = {
                    "channel": channel,
                    "animal_id": animal_id,
                    "sex": sex,
                    "group": group,
                    "subrow_id": len(subrows),
                    # Inherit from parent
                    "protocol": f.get("protocol", ""),
                    "stim_type": f.get("stim_type", ""),
                    "power": f.get("power", ""),
                    "experiment": f.get("experiment", ""),
                    "strain": f.get("strain", ""),
                }
                subrow.update(extra)
                subrows.append(subrow)
                self._dirty = True
                return subrow
        return None

    def get_subrows(self, file_path: str) -> List[Dict[str, Any]]:
        """Get subrows for a file."""
        file_path_resolved = str(Path(self._to_absolute(file_path)).resolve())
        for f in self._files:
            if str(Path(f.get("file_path", "")).resolve()) == file_path_resolved:
                return list(f.get("subrows", []))
        return []

    def update_subrow(
        self, file_path: str, subrow_id: int, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update a specific subrow by index."""
        file_path_resolved = str(Path(self._to_absolute(file_path)).resolve())
        for f in self._files:
            if str(Path(f.get("file_path", "")).resolve()) == file_path_resolved:
                subrows = f.get("subrows", [])
                if 0 <= subrow_id < len(subrows):
                    for key, value in updates.items():
                        if key in EDITABLE_COLUMNS or key in ("channel", "group", "subrow_id"):
                            subrows[subrow_id][key] = value
                    self._dirty = True
                    return dict(subrows[subrow_id])
        return None

    # ------------------------------------------------------------------
    # Provenance tracking
    # ------------------------------------------------------------------

    def _record_provenance(
        self,
        file_path: str,
        field: str,
        value: str,
        source_type: str,
        source_detail: str = "",
        source_preview: str = "",
        confidence: float = 0.5,
        reason: str = "",
    ) -> Optional[int]:
        """Record provenance for a metadata field value. Returns prov_id or None."""
        if self.cache is None:
            return None
        return self.cache.add_provenance(
            file_path=file_path,
            field=field,
            value=value,
            source_type=source_type,
            source_detail=source_detail,
            source_preview=source_preview,
            confidence=confidence,
            reason=reason,
        )

    def get_provenance(
        self, file_path: str, field: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get provenance records for a file (optionally filtered by field)."""
        if self.cache is None:
            return []
        rel_path = self._to_relative(file_path)
        return self.cache.get_provenance(rel_path, field)

    # ------------------------------------------------------------------
    # Enhanced ABF reference extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_abf_references(text: str) -> Set[str]:
        """
        Extract ABF file stem references from text.

        Supports multiple reference styles:
        - Full filename: "2024_03_21_0017.abf"
        - 6-10 digit numbers: "25708007"
        - Date_sequence patterns: "2024_03_21_0017"
        - Short numeric references: 3-5 digit numbers near ABF context
        """
        refs = set()

        # Match .abf file references (with or without extension)
        abf_matches = re.findall(r"(\w+)\.abf", text, re.IGNORECASE)
        refs.update(abf_matches)

        # Match underscore-separated date_sequence patterns (common ABF naming)
        date_seq = re.findall(r"\b(\d{4}_\d{2}_\d{2}_\d{4})\b", text)
        refs.update(date_seq)

        # Match 6-10 digit numbers (common ABF file naming)
        num_matches = re.findall(r"\b(\d{6,10})\b", text)
        refs.update(num_matches)

        # Match short numeric refs (3-5 digits) only if near ABF-related context
        # These could be shorthand like "017" or "17" referring to file suffixes
        short_matches = re.findall(r"(?:file|abf|recording|#)\s*(\d{2,5})\b", text, re.IGNORECASE)
        refs.update(short_matches)

        # Match sequential ranges like "files 5-8" or "recordings 10-15"
        range_matches = re.findall(r"(?:file|recording|abf)s?\s+(\d+)\s*[-–]\s*(\d+)", text, re.IGNORECASE)
        for start, end in range_matches:
            try:
                s, e = int(start), int(end)
                if e - s < 50:  # Sanity limit
                    for i in range(s, e + 1):
                        refs.add(str(i).zfill(len(start)))
            except ValueError:
                pass

        return refs

    # ------------------------------------------------------------------
    # Notes file discovery (legacy compat)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # File hash for caching
    # ------------------------------------------------------------------

    @staticmethod
    def file_hash(path: Path) -> str:
        """Compute SHA-256 hash of a file for change detection."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
