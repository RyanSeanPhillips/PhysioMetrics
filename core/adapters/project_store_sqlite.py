"""
SQLite primary store for project metadata.

Replaces in-memory list + JSON as the source of truth for project files,
metadata, subrows, provenance, custom columns, and analyses.

DB location: %APPDATA%/PhysioMetrics/PhysioMetrics.db (local, never on network drive)
JSON on network drive is a portable export, auto-merged on open.

WAL mode for concurrent reads. Pure Python — no PyQt6 imports.
"""

import sqlite3
import json
import hashlib
import shutil
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from contextlib import contextmanager


SCHEMA_VERSION = 1

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS projects (
    project_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT NOT NULL,
    data_directory TEXT NOT NULL UNIQUE,
    json_path      TEXT,
    json_mtime     REAL DEFAULT 0,
    json_hash      TEXT DEFAULT '',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    file_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id     INTEGER NOT NULL REFERENCES projects ON DELETE CASCADE,
    file_path      TEXT NOT NULL,
    file_name      TEXT NOT NULL,
    file_type      TEXT DEFAULT '',
    protocol       TEXT DEFAULT '',
    channel_count  INTEGER DEFAULT 0,
    sweep_count    INTEGER DEFAULT 0,
    keywords_display TEXT DEFAULT '',
    experiment     TEXT DEFAULT '',
    strain         TEXT DEFAULT '',
    stim_type      TEXT DEFAULT '',
    power          TEXT DEFAULT '',
    sex            TEXT DEFAULT '',
    animal_id      TEXT DEFAULT '',
    channel        TEXT DEFAULT '',
    stim_channel   TEXT DEFAULT '',
    events_channel TEXT DEFAULT '',
    status         TEXT DEFAULT 'pending',
    tags           TEXT DEFAULT '',
    notes          TEXT DEFAULT '',
    group_name     TEXT DEFAULT '',
    weight         TEXT DEFAULT '',
    age            TEXT DEFAULT '',
    date_recorded  TEXT DEFAULT '',
    linked_notes   TEXT DEFAULT '',
    updated_at     TEXT NOT NULL,
    field_timestamps TEXT DEFAULT '{}',
    UNIQUE(project_id, file_path)
);

CREATE INDEX IF NOT EXISTS idx_files_project ON files(project_id);
CREATE INDEX IF NOT EXISTS idx_files_animal ON files(project_id, animal_id);
CREATE INDEX IF NOT EXISTS idx_files_experiment ON files(project_id, experiment);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(project_id, status);

CREATE TABLE IF NOT EXISTS subrows (
    subrow_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id        INTEGER NOT NULL REFERENCES files ON DELETE CASCADE,
    subrow_index   INTEGER NOT NULL,
    channel        TEXT DEFAULT '',
    animal_id      TEXT DEFAULT '',
    sex            TEXT DEFAULT '',
    group_name     TEXT DEFAULT '',
    protocol       TEXT DEFAULT '',
    stim_type      TEXT DEFAULT '',
    power          TEXT DEFAULT '',
    experiment     TEXT DEFAULT '',
    strain         TEXT DEFAULT '',
    updated_at     TEXT NOT NULL,
    UNIQUE(file_id, subrow_index)
);

CREATE TABLE IF NOT EXISTS custom_columns (
    column_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id     INTEGER NOT NULL REFERENCES projects ON DELETE CASCADE,
    column_key     TEXT NOT NULL,
    display_name   TEXT NOT NULL,
    column_type    TEXT DEFAULT 'text',
    sort_order     INTEGER DEFAULT 0,
    UNIQUE(project_id, column_key)
);

CREATE TABLE IF NOT EXISTS custom_values (
    file_id        INTEGER NOT NULL REFERENCES files ON DELETE CASCADE,
    column_key     TEXT NOT NULL,
    value          TEXT DEFAULT '',
    updated_at     TEXT NOT NULL,
    PRIMARY KEY (file_id, column_key)
);

CREATE TABLE IF NOT EXISTS provenance (
    prov_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id        INTEGER NOT NULL REFERENCES files ON DELETE CASCADE,
    field          TEXT NOT NULL,
    value          TEXT NOT NULL,
    source_type    TEXT NOT NULL,
    source_detail  TEXT DEFAULT '',
    source_preview TEXT DEFAULT '',
    confidence     REAL DEFAULT 0.5,
    reason         TEXT DEFAULT '',
    supersedes     INTEGER REFERENCES provenance(prov_id),
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prov_file_field ON provenance(file_id, field);

CREATE TABLE IF NOT EXISTS notes_sources (
    note_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id     INTEGER NOT NULL REFERENCES projects ON DELETE CASCADE,
    file_path      TEXT NOT NULL,
    file_hash      TEXT NOT NULL,
    content        TEXT NOT NULL,
    abf_refs       TEXT DEFAULT '[]',
    parsed_at      TEXT NOT NULL,
    UNIQUE(project_id, file_hash)
);

CREATE TABLE IF NOT EXISTS analyses (
    analysis_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id        INTEGER NOT NULL REFERENCES files ON DELETE CASCADE,
    analysis_type  TEXT NOT NULL,
    output_path    TEXT DEFAULT '',
    parameters     TEXT DEFAULT '{}',
    status         TEXT DEFAULT 'completed',
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_analyses_file ON analyses(file_id);

CREATE TABLE IF NOT EXISTS backups (
    backup_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    backup_path    TEXT NOT NULL,
    trigger_event  TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_version (
    version        INTEGER PRIMARY KEY,
    applied_at     TEXT NOT NULL
);
"""

# Standard file columns that map directly to DB columns
STANDARD_FILE_COLUMNS = {
    'file_path', 'file_name', 'file_type', 'protocol', 'channel_count',
    'sweep_count', 'keywords_display', 'experiment', 'strain', 'stim_type',
    'power', 'sex', 'animal_id', 'channel', 'stim_channel', 'events_channel',
    'status', 'tags', 'notes', 'group_name', 'weight', 'age',
    'date_recorded', 'linked_notes',
}

# Map between dict key names and DB column names where they differ
_KEY_TO_COL = {'group': 'group_name'}
_COL_TO_KEY = {'group_name': 'group'}


def _db_path() -> Path:
    """Get the central DB path in %APPDATA%/PhysioMetrics/."""
    appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    db_dir = appdata / "PhysioMetrics"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "PhysioMetrics.db"


def _backup_dir() -> Path:
    """Get the backup directory."""
    d = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "PhysioMetrics" / "backups"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now() -> str:
    return datetime.now().isoformat()


def _dict_key(col: str) -> str:
    """Convert DB column name to dict key name."""
    return _COL_TO_KEY.get(col, col)


def _db_col(key: str) -> str:
    """Convert dict key name to DB column name."""
    return _KEY_TO_COL.get(key, key)


class ProjectStoreSQLite:
    """
    Central SQLite store for all project metadata.

    One DB at %APPDATA%/PhysioMetrics/PhysioMetrics.db holds all projects.
    JSON files on network drives are portable exports, merged on open.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else _db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; use explicit transactions
        )
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist and record schema version."""
        self._conn.executescript(SCHEMA_SQL)
        # Record schema version if not present
        existing = self._conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
        if existing is None:
            self._conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, _now()),
            )

    @contextmanager
    def transaction(self):
        """Explicit transaction context manager."""
        self._conn.execute("BEGIN")
        try:
            yield
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------
    # Projects CRUD
    # ------------------------------------------------------------------

    def upsert_project(
        self, name: str, data_directory: str,
        json_path: Optional[str] = None,
    ) -> int:
        """Create or update a project. Returns project_id."""
        now = _now()
        # Normalize the data directory for consistent matching
        norm_dir = str(Path(data_directory).resolve())

        existing = self._conn.execute(
            "SELECT project_id FROM projects WHERE data_directory = ?",
            (norm_dir,),
        ).fetchone()

        if existing:
            pid = existing["project_id"]
            self._conn.execute(
                """UPDATE projects SET name=?, json_path=?, updated_at=?
                   WHERE project_id=?""",
                (name, json_path, now, pid),
            )
            return pid
        else:
            cursor = self._conn.execute(
                """INSERT INTO projects (name, data_directory, json_path, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, norm_dir, json_path, now, now),
            )
            return cursor.lastrowid

    def get_project(self, project_id: int) -> Optional[Dict[str, Any]]:
        """Get project by ID."""
        row = self._conn.execute(
            "SELECT * FROM projects WHERE project_id = ?", (project_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_project_by_directory(self, data_directory: str) -> Optional[Dict[str, Any]]:
        """Find project by data directory path."""
        norm_dir = str(Path(data_directory).resolve())
        row = self._conn.execute(
            "SELECT * FROM projects WHERE data_directory = ?", (norm_dir,)
        ).fetchone()
        return dict(row) if row else None

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects."""
        rows = self._conn.execute(
            "SELECT * FROM projects ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_project_json_sync(
        self, project_id: int, json_mtime: float, json_hash: str
    ):
        """Update the JSON sync metadata after a successful sync."""
        self._conn.execute(
            """UPDATE projects SET json_mtime=?, json_hash=?, updated_at=?
               WHERE project_id=?""",
            (json_mtime, json_hash, _now(), project_id),
        )

    # ------------------------------------------------------------------
    # Files CRUD
    # ------------------------------------------------------------------

    def upsert_file(
        self, project_id: int, file_path: str, data: Dict[str, Any]
    ) -> int:
        """Insert or update a file record. Returns file_id."""
        now = _now()

        existing = self._conn.execute(
            "SELECT file_id FROM files WHERE project_id = ? AND file_path = ?",
            (project_id, file_path),
        ).fetchone()

        # Separate standard columns from custom
        std_data = {}
        custom_data = {}
        for key, value in data.items():
            db_col = _db_col(key)
            if db_col in STANDARD_FILE_COLUMNS:
                std_data[db_col] = value
            elif key not in ('file_id', 'project_id', 'updated_at',
                             'field_timestamps', 'subrows', 'exports',
                             'path_keywords', 'stim_channels', 'stim_frequency'):
                # Custom field
                custom_data[key] = str(value) if value is not None else ''

        # Ensure file_path and file_name are set
        std_data.setdefault('file_path', file_path)
        std_data.setdefault('file_name', Path(file_path).name)

        if existing:
            fid = existing["file_id"]
            # Build UPDATE SET clause for standard columns
            if std_data:
                set_parts = [f"{col} = ?" for col in std_data.keys()]
                set_parts.append("updated_at = ?")
                vals = list(std_data.values()) + [now, fid]
                self._conn.execute(
                    f"UPDATE files SET {', '.join(set_parts)} WHERE file_id = ?",
                    vals,
                )
        else:
            # INSERT
            cols = list(std_data.keys()) + ['project_id', 'updated_at']
            placeholders = ', '.join(['?'] * len(cols))
            vals = list(std_data.values()) + [project_id, now]
            cursor = self._conn.execute(
                f"INSERT INTO files ({', '.join(cols)}) VALUES ({placeholders})",
                vals,
            )
            fid = cursor.lastrowid

        # Upsert custom values
        for key, value in custom_data.items():
            self._conn.execute(
                """INSERT INTO custom_values (file_id, column_key, value, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(file_id, column_key) DO UPDATE SET
                     value=excluded.value, updated_at=excluded.updated_at""",
                (fid, key, value, now),
            )

        return fid

    def get_file(self, file_id: int) -> Optional[Dict[str, Any]]:
        """Get a single file by ID, including custom values."""
        row = self._conn.execute(
            "SELECT * FROM files WHERE file_id = ?", (file_id,)
        ).fetchone()
        if not row:
            return None
        return self._file_row_to_dict(row)

    def get_file_by_path(self, project_id: int, file_path: str) -> Optional[Dict[str, Any]]:
        """Get a file by its relative path within a project."""
        row = self._conn.execute(
            "SELECT * FROM files WHERE project_id = ? AND file_path = ?",
            (project_id, file_path),
        ).fetchone()
        if not row:
            return None
        return self._file_row_to_dict(row)

    def get_file_id_by_path(self, project_id: int, file_path: str) -> Optional[int]:
        """Get just the file_id for a path (fast lookup)."""
        row = self._conn.execute(
            "SELECT file_id FROM files WHERE project_id = ? AND file_path = ?",
            (project_id, file_path),
        ).fetchone()
        return row["file_id"] if row else None

    def get_project_files(
        self, project_id: int,
        filter_field: Optional[str] = None,
        filter_value: Optional[str] = None,
        order_by: Optional[str] = None,
        offset: int = 0,
        limit: int = 10000,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get all files for a project with optional filtering.

        Returns (files_list, total_count).
        """
        where = "WHERE project_id = ?"
        params: list = [project_id]

        if filter_field and filter_value is not None:
            col = _db_col(filter_field)
            if col in STANDARD_FILE_COLUMNS:
                where += f" AND {col} = ?"
                params.append(filter_value)

        # Count total
        total = self._conn.execute(
            f"SELECT COUNT(*) FROM files {where}", params
        ).fetchone()[0]

        # Order
        order = "ORDER BY file_id"
        if order_by:
            col = _db_col(order_by)
            if col in STANDARD_FILE_COLUMNS:
                order = f"ORDER BY {col}"

        rows = self._conn.execute(
            f"SELECT * FROM files {where} {order} LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()

        files = [self._file_row_to_dict(r) for r in rows]
        return files, total

    def update_file(
        self, file_id: int, updates: Dict[str, Any],
        record_field_timestamps: bool = True,
    ) -> bool:
        """
        Update specific fields on a file.

        Accepts any field — standard fields update columns directly,
        unknown fields go to custom_values.
        """
        now = _now()
        std_updates = {}
        custom_updates = {}

        for key, value in updates.items():
            db_col = _db_col(key)
            if db_col in STANDARD_FILE_COLUMNS:
                std_updates[db_col] = value
            elif key not in ('file_id', 'project_id', 'updated_at', 'field_timestamps'):
                custom_updates[key] = str(value) if value is not None else ''

        if std_updates:
            # Update field timestamps for merge tracking
            if record_field_timestamps:
                existing = self._conn.execute(
                    "SELECT field_timestamps FROM files WHERE file_id = ?",
                    (file_id,),
                ).fetchone()
                if existing:
                    ts_map = json.loads(existing["field_timestamps"] or '{}')
                    for col in std_updates:
                        ts_map[_dict_key(col)] = now
                    std_updates['field_timestamps'] = json.dumps(ts_map)

            set_parts = [f"{col} = ?" for col in std_updates.keys()]
            set_parts.append("updated_at = ?")
            vals = list(std_updates.values()) + [now, file_id]
            self._conn.execute(
                f"UPDATE files SET {', '.join(set_parts)} WHERE file_id = ?",
                vals,
            )

        for key, value in custom_updates.items():
            self._conn.execute(
                """INSERT INTO custom_values (file_id, column_key, value, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(file_id, column_key) DO UPDATE SET
                     value=excluded.value, updated_at=excluded.updated_at""",
                (file_id, key, value, now),
            )

        return True

    def delete_file(self, file_id: int) -> bool:
        """Delete a file and all related records (CASCADE)."""
        cursor = self._conn.execute(
            "DELETE FROM files WHERE file_id = ?", (file_id,)
        )
        return cursor.rowcount > 0

    def _file_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a files table row to a dict, including custom values."""
        d = {}
        for key in row.keys():
            if key == 'field_timestamps':
                continue  # Internal; don't expose
            val = row[key]
            out_key = _dict_key(key)
            d[out_key] = val

        # Attach custom values
        file_id = row["file_id"]
        custom_rows = self._conn.execute(
            "SELECT column_key, value FROM custom_values WHERE file_id = ?",
            (file_id,),
        ).fetchall()
        for cr in custom_rows:
            d[cr["column_key"]] = cr["value"]

        # Attach subrow count
        subrow_count = self._conn.execute(
            "SELECT COUNT(*) FROM subrows WHERE file_id = ?", (file_id,)
        ).fetchone()[0]
        if subrow_count > 0:
            d['subrow_count'] = subrow_count

        return d

    # ------------------------------------------------------------------
    # Subrows
    # ------------------------------------------------------------------

    def add_subrow(
        self, file_id: int, subrow_data: Dict[str, Any]
    ) -> int:
        """Add a subrow to a file. Returns subrow_id."""
        now = _now()
        # Get next subrow_index
        max_idx = self._conn.execute(
            "SELECT COALESCE(MAX(subrow_index), -1) FROM subrows WHERE file_id = ?",
            (file_id,),
        ).fetchone()[0]
        idx = max_idx + 1

        cols = ['file_id', 'subrow_index', 'updated_at']
        vals = [file_id, idx, now]

        for key in ('channel', 'animal_id', 'sex', 'group_name', 'protocol',
                     'stim_type', 'power', 'experiment', 'strain'):
            dict_key = _dict_key(key)
            val = subrow_data.get(dict_key, subrow_data.get(key, ''))
            cols.append(key)
            vals.append(str(val))

        placeholders = ', '.join(['?'] * len(cols))
        cursor = self._conn.execute(
            f"INSERT INTO subrows ({', '.join(cols)}) VALUES ({placeholders})",
            vals,
        )
        return cursor.lastrowid

    def get_subrows(self, file_id: int) -> List[Dict[str, Any]]:
        """Get all subrows for a file."""
        rows = self._conn.execute(
            "SELECT * FROM subrows WHERE file_id = ? ORDER BY subrow_index",
            (file_id,),
        ).fetchall()
        result = []
        for r in rows:
            d = {}
            for key in r.keys():
                d[_dict_key(key)] = r[key]
            result.append(d)
        return result

    def update_subrow(self, subrow_id: int, updates: Dict[str, Any]) -> bool:
        """Update a subrow."""
        now = _now()
        set_parts = []
        vals = []
        for key, value in updates.items():
            col = _db_col(key)
            if col in ('channel', 'animal_id', 'sex', 'group_name', 'protocol',
                        'stim_type', 'power', 'experiment', 'strain'):
                set_parts.append(f"{col} = ?")
                vals.append(str(value))
        if not set_parts:
            return False
        set_parts.append("updated_at = ?")
        vals.extend([now, subrow_id])
        self._conn.execute(
            f"UPDATE subrows SET {', '.join(set_parts)} WHERE subrow_id = ?",
            vals,
        )
        return True

    # ------------------------------------------------------------------
    # Custom columns
    # ------------------------------------------------------------------

    def add_custom_column(
        self, project_id: int, column_key: str,
        display_name: str, column_type: str = 'text',
        sort_order: int = 0,
    ) -> int:
        """Define a custom metadata column for a project. Returns column_id."""
        cursor = self._conn.execute(
            """INSERT INTO custom_columns (project_id, column_key, display_name, column_type, sort_order)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(project_id, column_key) DO UPDATE SET
                 display_name=excluded.display_name,
                 column_type=excluded.column_type,
                 sort_order=excluded.sort_order""",
            (project_id, column_key, display_name, column_type, sort_order),
        )
        return cursor.lastrowid

    def get_custom_columns(self, project_id: int) -> List[Dict[str, Any]]:
        """Get custom column definitions for a project."""
        rows = self._conn.execute(
            "SELECT * FROM custom_columns WHERE project_id = ? ORDER BY sort_order",
            (project_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def add_provenance(
        self, file_id: int, field: str, value: str,
        source_type: str, source_detail: str = '',
        source_preview: str = '', confidence: float = 0.5,
        reason: str = '',
    ) -> int:
        """Record provenance for a metadata value. Returns prov_id."""
        # Find previous record for correction chain
        prev = self._conn.execute(
            """SELECT prov_id FROM provenance
               WHERE file_id = ? AND field = ?
               ORDER BY prov_id DESC LIMIT 1""",
            (file_id, field),
        ).fetchone()
        supersedes = prev["prov_id"] if prev else None

        cursor = self._conn.execute(
            """INSERT INTO provenance
               (file_id, field, value, source_type, source_detail, source_preview,
                confidence, reason, supersedes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (file_id, field, value, source_type, source_detail, source_preview,
             confidence, reason, supersedes, _now()),
        )
        return cursor.lastrowid

    def get_provenance(
        self, file_id: int, field: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get provenance records for a file, optionally filtered by field."""
        if field:
            rows = self._conn.execute(
                """SELECT * FROM provenance
                   WHERE file_id = ? AND field = ?
                   ORDER BY prov_id DESC""",
                (file_id, field),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM provenance WHERE file_id = ?
                   ORDER BY prov_id DESC""",
                (file_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Notes sources cache
    # ------------------------------------------------------------------

    def save_notes_cache(
        self, project_id: int, path: str, file_hash: str,
        content: Any, abf_refs: List[str],
    ) -> None:
        """Cache parsed notes content keyed by project + file hash."""
        self._conn.execute(
            """INSERT INTO notes_sources
               (project_id, file_path, file_hash, content, abf_refs, parsed_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(project_id, file_hash) DO UPDATE SET
                 file_path=excluded.file_path,
                 content=excluded.content,
                 abf_refs=excluded.abf_refs,
                 parsed_at=excluded.parsed_at""",
            (project_id, path, file_hash,
             json.dumps(content, default=str), json.dumps(abf_refs), _now()),
        )

    def get_notes_cache(self, project_id: int, file_hash: str) -> Optional[Dict]:
        """Get cached notes by file hash."""
        row = self._conn.execute(
            "SELECT * FROM notes_sources WHERE project_id = ? AND file_hash = ?",
            (project_id, file_hash),
        ).fetchone()
        if row:
            return {
                "file_path": row["file_path"],
                "content": json.loads(row["content"]),
                "abf_refs": json.loads(row["abf_refs"]),
                "parsed_at": row["parsed_at"],
            }
        return None

    # ------------------------------------------------------------------
    # Analyses
    # ------------------------------------------------------------------

    def record_analysis(
        self, file_id: int, analysis_type: str,
        output_path: str = '', parameters: Optional[Dict] = None,
        status: str = 'completed',
    ) -> int:
        """Record an analysis output for a file. Returns analysis_id."""
        cursor = self._conn.execute(
            """INSERT INTO analyses (file_id, analysis_type, output_path, parameters, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (file_id, analysis_type, output_path,
             json.dumps(parameters or {}), status, _now()),
        )
        return cursor.lastrowid

    def get_analyses(self, file_id: int) -> List[Dict[str, Any]]:
        """Get all analyses for a file."""
        rows = self._conn.execute(
            "SELECT * FROM analyses WHERE file_id = ? ORDER BY created_at DESC",
            (file_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Unique values & completeness
    # ------------------------------------------------------------------

    def get_unique_values(self, project_id: int, field: str) -> List[str]:
        """Get sorted unique non-empty values for a standard field."""
        col = _db_col(field)
        if col not in STANDARD_FILE_COLUMNS:
            # Try custom values
            rows = self._conn.execute(
                """SELECT DISTINCT cv.value FROM custom_values cv
                   JOIN files f ON f.file_id = cv.file_id
                   WHERE f.project_id = ? AND cv.column_key = ?
                     AND cv.value != '' AND cv.value IS NOT NULL
                   ORDER BY cv.value""",
                (project_id, field),
            ).fetchall()
            return [r[0] for r in rows]

        rows = self._conn.execute(
            f"""SELECT DISTINCT {col} FROM files
                WHERE project_id = ? AND {col} != '' AND {col} IS NOT NULL
                  AND {col} NOT IN ('Loading...', 'Unknown')
                ORDER BY {col}""",
            (project_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_metadata_completeness(self, project_id: int) -> Dict[str, Any]:
        """Get fill percentage for metadata columns."""
        total = self._conn.execute(
            "SELECT COUNT(*) FROM files WHERE project_id = ?", (project_id,)
        ).fetchone()[0]
        if total == 0:
            return {"per_column": {}, "overall_pct": 0.0, "total_files": 0}

        fields = ["experiment", "strain", "stim_type", "power", "sex",
                   "animal_id", "channel", "stim_channel", "protocol"]

        per_column = {}
        total_filled = 0
        total_cells = 0

        for field in fields:
            col = _db_col(field)
            filled = self._conn.execute(
                f"""SELECT COUNT(*) FROM files
                    WHERE project_id = ? AND {col} != '' AND {col} IS NOT NULL
                      AND {col} NOT IN ('Loading...', 'Unknown', 'pending')""",
                (project_id,),
            ).fetchone()[0]
            pct = (filled / total) * 100
            per_column[field] = {
                "filled": filled,
                "total": total,
                "percent": round(pct, 1),
            }
            total_filled += filled
            total_cells += total

        overall = (total_filled / total_cells * 100) if total_cells > 0 else 0
        return {
            "per_column": per_column,
            "overall_pct": round(overall, 1),
            "total_files": total,
        }

    # ------------------------------------------------------------------
    # JSON import / export (for network-drive .physiometrics files)
    # ------------------------------------------------------------------

    def export_to_json(self, project_id: int) -> Dict[str, Any]:
        """
        Export a project to a JSON-serializable dict (for .physiometrics file).

        Format compatible with ProjectManager.load_project().
        """
        project = self.get_project(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        files, _ = self.get_project_files(project_id)

        # Convert to the JSON format expected by ProjectManager
        files_json = []
        for f in files:
            entry = dict(f)
            # Remove internal IDs
            entry.pop('file_id', None)
            entry.pop('project_id', None)
            entry.pop('subrow_count', None)
            # Rename group back
            if 'group' in entry and 'group_name' not in entry:
                pass  # Already correct from _file_row_to_dict
            # Flatten subrows into separate file entries (app expects flat format)
            file_id = f.get('file_id')
            if file_id:
                subrows = self.get_subrows(file_id)
                if subrows:
                    for sr in subrows:
                        sr.pop('subrow_id', None)
                        sr.pop('file_id', None)
                    # Keep nested subrows for round-trip
                    entry['subrows'] = subrows
                    # Also emit flat entries for the app table
                    for sr in subrows:
                        flat = dict(entry)
                        flat.pop('subrows', None)
                        flat.pop('subrow_count', None)
                        # Override with subrow-specific fields
                        for sk in ('channel', 'animal_id', 'sex', 'group',
                                   'protocol', 'stim_type', 'power',
                                   'experiment', 'strain'):
                            if sk in sr and sr[sk]:
                                flat[sk] = sr[sk]
                        flat['is_sub_row'] = True
                        flat['parent_file'] = entry.get('file_path', '')
                        files_json.append(flat)
            files_json.append(entry)

        # Custom columns
        custom_cols = self.get_custom_columns(project_id)

        return {
            "version": 2,
            "project_name": project["name"],
            "data_directory": ".",
            "created": project["created_at"],
            "last_modified": _now(),
            "file_count": len(files_json),
            "files": files_json,
            "experiments": [],
            "notes_files": [],
            "custom_columns": [
                {"key": c["column_key"], "name": c["display_name"],
                 "type": c["column_type"]}
                for c in custom_cols
            ],
        }

    def import_from_json(
        self, project_id: int, json_data: Dict[str, Any],
        data_directory: Path,
    ) -> Dict[str, Any]:
        """
        Import files from a JSON dict (loaded .physiometrics) into the DB.

        This is a non-destructive merge — existing DB records are kept.
        New files from JSON are added. Changed fields use field_timestamps
        to resolve conflicts.

        Returns:
            Merge report: {accepted: N, conflicts: [...], new: N, unchanged: N}
        """
        now = _now()
        files_data = json_data.get("files", [])

        accepted = 0
        new_count = 0
        conflicts = []
        unchanged = 0

        for f_data in files_data:
            # Skip flat subrow entries (they're reconstructed from nested subrows)
            if f_data.get("is_sub_row"):
                continue

            # Normalize file_path to relative
            file_path = str(f_data.get("file_path", ""))
            if not file_path:
                continue

            # Make path relative if absolute
            try:
                p = Path(file_path)
                if p.is_absolute():
                    file_path = str(p.relative_to(data_directory))
            except (ValueError, TypeError):
                pass
            file_path = file_path.replace("\\", "/")

            existing = self._conn.execute(
                "SELECT file_id, field_timestamps, updated_at FROM files WHERE project_id = ? AND file_path = ?",
                (project_id, file_path),
            ).fetchone()

            if existing is None:
                # New file — insert from JSON
                clean = dict(f_data)
                clean['file_path'] = file_path
                clean.pop('subrows', None)
                clean.pop('exports', None)
                self.upsert_file(project_id, file_path, clean)

                # Import subrows if present
                fid = self.get_file_id_by_path(project_id, file_path)
                for sr in f_data.get('subrows', []):
                    if fid:
                        self.add_subrow(fid, sr)

                new_count += 1
            else:
                # Existing file — merge fields
                fid = existing["file_id"]
                db_timestamps = json.loads(existing["field_timestamps"] or '{}')

                file_conflicts = []
                updates_to_apply = {}

                for key, json_val in f_data.items():
                    if key in ('file_path', 'file_name', 'subrows', 'exports',
                               'path_keywords', 'stim_channels', 'stim_frequency',
                               'file_id', 'project_id', 'updated_at', 'field_timestamps'):
                        continue

                    json_val_str = str(json_val) if json_val is not None else ''
                    if not json_val_str or json_val_str in ('', 'Loading...', 'Unknown'):
                        continue

                    # Check if DB field was modified since last sync
                    db_field_ts = db_timestamps.get(key)
                    last_sync = existing["updated_at"]  # Approximate

                    if db_field_ts is None:
                        # DB field never explicitly set — accept JSON value
                        updates_to_apply[key] = json_val_str
                        accepted += 1
                    else:
                        # DB field was modified — check current DB value
                        col = _db_col(key)
                        if col in STANDARD_FILE_COLUMNS:
                            db_row = self._conn.execute(
                                f"SELECT {col} FROM files WHERE file_id = ?",
                                (fid,),
                            ).fetchone()
                            db_val = str(db_row[0]) if db_row and db_row[0] else ''
                        else:
                            cv = self._conn.execute(
                                "SELECT value FROM custom_values WHERE file_id = ? AND column_key = ?",
                                (fid, key),
                            ).fetchone()
                            db_val = cv["value"] if cv else ''

                        if db_val == json_val_str:
                            unchanged += 1
                        elif not db_val:
                            updates_to_apply[key] = json_val_str
                            accepted += 1
                        else:
                            # Both modified with different values — conflict
                            file_conflicts.append({
                                "field": key,
                                "db_value": db_val,
                                "json_value": json_val_str,
                            })

                if updates_to_apply:
                    self.update_file(fid, updates_to_apply, record_field_timestamps=False)

                if file_conflicts:
                    conflicts.append({
                        "file_path": file_path,
                        "conflicts": file_conflicts,
                    })

        # Import custom column definitions if present
        for cc in json_data.get("custom_columns", []):
            self.add_custom_column(
                project_id, cc["key"], cc.get("name", cc["key"]),
                cc.get("type", "text"),
            )

        return {
            "accepted": accepted,
            "new": new_count,
            "conflicts": conflicts,
            "unchanged": unchanged,
        }

    # ------------------------------------------------------------------
    # Backup & restore
    # ------------------------------------------------------------------

    def backup(self, trigger_event: str = "manual") -> str:
        """
        Create a backup of the database using SQLite's backup API.

        Returns the backup file path.
        """
        backup_path = _backup_dir() / f"PhysioMetrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"

        # Use SQLite backup API for atomic, consistent snapshot
        backup_conn = sqlite3.connect(str(backup_path))
        try:
            self._conn.backup(backup_conn)
        finally:
            backup_conn.close()

        # Record in backups table
        self._conn.execute(
            "INSERT INTO backups (backup_path, trigger_event, created_at) VALUES (?, ?, ?)",
            (str(backup_path), trigger_event, _now()),
        )

        # Prune old backups
        self._prune_backups()

        return str(backup_path)

    def _prune_backups(self, keep: int = 10):
        """Keep only the most recent N backups, delete older ones."""
        rows = self._conn.execute(
            "SELECT backup_id, backup_path FROM backups ORDER BY created_at DESC"
        ).fetchall()

        for row in rows[keep:]:
            try:
                bp = Path(row["backup_path"])
                if bp.exists():
                    bp.unlink()
            except OSError:
                pass
            self._conn.execute(
                "DELETE FROM backups WHERE backup_id = ?", (row["backup_id"],)
            )

    def list_backups(self) -> List[Dict[str, Any]]:
        """List all backups."""
        rows = self._conn.execute(
            "SELECT * FROM backups ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Import from legacy ProjectCacheSQLite
    # ------------------------------------------------------------------

    def import_legacy_provenance(
        self, project_id: int, legacy_db_path: Path,
        path_to_file_id: Dict[str, int],
    ) -> int:
        """
        Import provenance records from legacy cache.db into the new store.

        Args:
            project_id: Target project.
            legacy_db_path: Path to old cache.db.
            path_to_file_id: Mapping of relative file_path -> file_id in new DB.

        Returns:
            Number of records imported.
        """
        if not legacy_db_path.exists():
            return 0

        legacy_conn = sqlite3.connect(str(legacy_db_path))
        legacy_conn.row_factory = sqlite3.Row

        try:
            rows = legacy_conn.execute(
                "SELECT * FROM pj_provenance ORDER BY prov_id"
            ).fetchall()
        except sqlite3.OperationalError:
            legacy_conn.close()
            return 0

        imported = 0
        for row in rows:
            file_path = row["file_path"]
            file_id = path_to_file_id.get(file_path)
            if file_id is None:
                continue

            self._conn.execute(
                """INSERT INTO provenance
                   (file_id, field, value, source_type, source_detail,
                    source_preview, confidence, reason, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (file_id, row["field"], row["value"], row["source_type"],
                 row["source_detail"], row["source_preview"],
                 row["confidence"], row["reason"], row["created_at"]),
            )
            imported += 1

        legacy_conn.close()
        return imported

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_file_count(self, project_id: int) -> int:
        """Get total number of files in a project."""
        return self._conn.execute(
            "SELECT COUNT(*) FROM files WHERE project_id = ?", (project_id,)
        ).fetchone()[0]

    @staticmethod
    def file_hash(path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
