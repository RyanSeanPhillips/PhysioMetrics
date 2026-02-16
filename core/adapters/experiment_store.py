"""
Experiment store — SQLite adapter for the v2 simplified schema.

Central DB at %APPDATA%/PhysioMetrics/PhysioMetrics.db.
One row per analyzable unit (file + channel + animal combination).
Replaces project_store_sqlite.py (v1 schema).

Pure Python — no PyQt6 imports.
"""

import sqlite3
import json
import hashlib
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from contextlib import contextmanager


SCHEMA_VERSION = 2

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- One row per analyzable unit (file + channel + animal combination)
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    -- File identity
    file_path        TEXT NOT NULL,
    file_name        TEXT NOT NULL,
    file_type        TEXT DEFAULT '',
    -- Per-animal identity
    animal_id        TEXT DEFAULT '',
    channel          TEXT DEFAULT '',
    -- Metadata
    experiment_name  TEXT DEFAULT '',
    strain           TEXT DEFAULT '',
    stim_type        TEXT DEFAULT '',
    power            TEXT DEFAULT '',
    sex              TEXT DEFAULT '',
    group_name       TEXT DEFAULT '',
    status           TEXT DEFAULT 'pending',
    -- File-level metadata
    protocol         TEXT DEFAULT '',
    channel_count    INTEGER DEFAULT 0,
    sweep_count      INTEGER DEFAULT 0,
    keywords_display TEXT DEFAULT '',
    stim_channel     TEXT DEFAULT '',
    events_channel   TEXT DEFAULT '',
    tags             TEXT DEFAULT '',
    notes            TEXT DEFAULT '',
    weight           TEXT DEFAULT '',
    age              TEXT DEFAULT '',
    date_recorded    TEXT DEFAULT '',
    -- Timestamps
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    -- Unique: one animal+channel per file
    UNIQUE(file_path, channel, animal_id)
);

CREATE INDEX IF NOT EXISTS idx_exp_animal ON experiments(animal_id);
CREATE INDEX IF NOT EXISTS idx_exp_file ON experiments(file_path);
CREATE INDEX IF NOT EXISTS idx_exp_name ON experiments(experiment_name);
CREATE INDEX IF NOT EXISTS idx_exp_strain ON experiments(strain);
CREATE INDEX IF NOT EXISTS idx_exp_status ON experiments(status);

-- Reference documents (notes files, spreadsheets, surgery logs, etc.)
CREATE TABLE IF NOT EXISTS sources (
    source_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path        TEXT NOT NULL UNIQUE,
    file_name        TEXT NOT NULL,
    file_type        TEXT DEFAULT '',
    file_hash        TEXT DEFAULT '',
    parsed_content   TEXT DEFAULT '',
    parsed_at        TEXT DEFAULT '',
    created_at       TEXT NOT NULL
);

-- Links between sources and experiments/animals
CREATE TABLE IF NOT EXISTS source_links (
    link_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id        INTEGER NOT NULL REFERENCES sources ON DELETE CASCADE,
    animal_id        TEXT NOT NULL,
    experiment_id    INTEGER REFERENCES experiments ON DELETE SET NULL,
    field            TEXT NOT NULL,
    value            TEXT NOT NULL,
    location         TEXT DEFAULT '',
    confidence       REAL DEFAULT 0.5,
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sl_animal ON source_links(animal_id);
CREATE INDEX IF NOT EXISTS idx_sl_experiment ON source_links(experiment_id);
CREATE INDEX IF NOT EXISTS idx_sl_source ON source_links(source_id);
CREATE INDEX IF NOT EXISTS idx_sl_field ON source_links(field);

-- Saved filter queries that export to JSON for sharing
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL,
    description      TEXT DEFAULT '',
    filter_query     TEXT NOT NULL DEFAULT '{}',
    data_directory   TEXT DEFAULT '',
    json_path        TEXT DEFAULT '',
    json_mtime       REAL DEFAULT 0,
    json_hash        TEXT DEFAULT '',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

-- Backup history
CREATE TABLE IF NOT EXISTS backups (
    backup_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    backup_path      TEXT NOT NULL,
    trigger_event    TEXT NOT NULL,
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_version (
    version          INTEGER PRIMARY KEY,
    applied_at       TEXT NOT NULL
);
"""

# FTS5 virtual table — standalone (no content= sync issues)
FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS experiments_fts USING fts5(
    file_name, experiment_name, strain, stim_type, animal_id,
    tags, notes, protocol
);
"""

# Standard experiment columns that map directly to DB columns
STANDARD_COLUMNS = {
    'file_path', 'file_name', 'file_type', 'animal_id', 'channel',
    'experiment_name', 'strain', 'stim_type', 'power', 'sex',
    'group_name', 'status', 'protocol', 'channel_count', 'sweep_count',
    'keywords_display', 'stim_channel', 'events_channel', 'tags',
    'notes', 'weight', 'age', 'date_recorded',
}

# Map between legacy dict key names and DB column names
_KEY_TO_COL = {
    'group': 'group_name',
    'experiment': 'experiment_name',
    'keywords': 'keywords_display',
}
_COL_TO_KEY = {v: k for k, v in _KEY_TO_COL.items()}


def _db_path() -> Path:
    """Get the central DB path in %APPDATA%/PhysioMetrics/."""
    appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    db_dir = appdata / "PhysioMetrics"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "PhysioMetrics.db"


def _backup_dir() -> Path:
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


class ExperimentStore:
    """
    Central SQLite store for all experiment metadata (v2 schema).

    One DB at %APPDATA%/PhysioMetrics/PhysioMetrics.db.
    Flat experiments table — no subrows, no EAV custom_values.
    Custom columns via ALTER TABLE ADD COLUMN.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else _db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist, auto-migrate v1->v2 if needed."""
        # Check if v1 tables exist and need migration
        try:
            tables = {r[0] for r in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            needs_migration = 'projects' in tables and 'files' in tables and 'experiments' not in tables
        except Exception:
            needs_migration = False

        # Create v2 schema
        self._conn.executescript(SCHEMA_SQL)
        try:
            self._conn.execute(FTS_SQL)
        except sqlite3.OperationalError:
            pass

        existing = self._conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
        if existing is None or existing["version"] < SCHEMA_VERSION:
            self._conn.execute(
                "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, _now()),
            )

        # Auto-migrate v1 data if v1 tables exist
        if needs_migration:
            try:
                from _internal.scripts.migrate_v1_to_v2 import migrate
                print("[experiment-store] Auto-migrating v1 data to v2 schema...")
                self._conn.close()
                migrate(self.db_path)
                self._conn = sqlite3.connect(
                    str(self.db_path), check_same_thread=False, isolation_level=None,
                )
                self._conn.row_factory = sqlite3.Row
            except Exception as e:
                print(f"[experiment-store] Auto-migration failed (non-fatal): {e}")

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
    # Experiments CRUD
    # ------------------------------------------------------------------

    def upsert_experiment(self, data: Dict[str, Any]) -> int:
        """Insert or update an experiment row. Returns experiment_id.

        Unique key: (file_path, channel, animal_id).
        """
        now = _now()

        # Separate standard from dynamic columns
        std_data = {}
        dyn_data = {}
        for key, value in data.items():
            db_col = _db_col(key)
            if db_col in STANDARD_COLUMNS:
                std_data[db_col] = value
            elif key not in ('experiment_id', 'created_at', 'updated_at',
                             'exports', 'path_keywords', 'stim_channels',
                             'stim_frequency', 'is_sub_row', 'parent_file',
                             'subrows', 'subrow_count', 'file_id', 'project_id',
                             'linked_notes', 'field_timestamps'):
                dyn_data[key] = str(value) if value is not None else ''

        file_path = std_data.get('file_path', '')
        channel = std_data.get('channel', '')
        animal_id = std_data.get('animal_id', '')
        std_data.setdefault('file_name', Path(file_path).name if file_path else '')

        # Check for existing row
        existing = self._conn.execute(
            "SELECT experiment_id FROM experiments WHERE file_path = ? AND channel = ? AND animal_id = ?",
            (file_path, channel, animal_id),
        ).fetchone()

        if existing:
            eid = existing["experiment_id"]
            if std_data:
                set_parts = [f"{col} = ?" for col in std_data.keys()]
                set_parts.append("updated_at = ?")
                vals = list(std_data.values()) + [now, eid]
                self._conn.execute(
                    f"UPDATE experiments SET {', '.join(set_parts)} WHERE experiment_id = ?",
                    vals,
                )
        else:
            cols = list(std_data.keys()) + ['created_at', 'updated_at']
            placeholders = ', '.join(['?'] * len(cols))
            vals = list(std_data.values()) + [now, now]
            cursor = self._conn.execute(
                f"INSERT INTO experiments ({', '.join(cols)}) VALUES ({placeholders})",
                vals,
            )
            eid = cursor.lastrowid

        # Update dynamic (custom) columns
        for key, value in dyn_data.items():
            self._ensure_column(key)
            self._conn.execute(
                f"UPDATE experiments SET {key} = ?, updated_at = ? WHERE experiment_id = ?",
                (value, now, eid),
            )

        # Update FTS index
        self._update_fts(eid)

        return eid

    def get_experiment(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Get a single experiment by ID."""
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_experiment_by_key(
        self, file_path: str, channel: str = '', animal_id: str = ''
    ) -> Optional[Dict[str, Any]]:
        """Get experiment by unique key (file_path, channel, animal_id)."""
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE file_path = ? AND channel = ? AND animal_id = ?",
            (file_path, channel, animal_id),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_experiments(
        self,
        filter_field: Optional[str] = None,
        filter_value: Optional[str] = None,
        order_by: Optional[str] = None,
        offset: int = 0,
        limit: int = 10000,
        **filters,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get experiments with optional filtering. Returns (list, total_count)."""
        where_parts = []
        params: list = []

        # Single field filter
        if filter_field and filter_value is not None:
            col = _db_col(filter_field)
            where_parts.append(f"{col} = ?")
            params.append(filter_value)

        # Additional keyword filters
        for key, value in filters.items():
            col = _db_col(key)
            where_parts.append(f"{col} = ?")
            params.append(value)

        where = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        total = self._conn.execute(
            f"SELECT COUNT(*) FROM experiments {where}", params
        ).fetchone()[0]

        order = "ORDER BY experiment_id"
        if order_by:
            col = _db_col(order_by)
            order = f"ORDER BY {col}"

        rows = self._conn.execute(
            f"SELECT * FROM experiments {where} {order} LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()

        return [self._row_to_dict(r) for r in rows], total

    def get_experiments_by_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Get all experiments for a given file."""
        rows = self._conn.execute(
            "SELECT * FROM experiments WHERE file_path = ? ORDER BY experiment_id",
            (file_path,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_experiments_for_animal(self, animal_id: str) -> List[Dict[str, Any]]:
        """Get all experiments for an animal across all files."""
        rows = self._conn.execute(
            "SELECT * FROM experiments WHERE animal_id = ? ORDER BY file_path",
            (animal_id,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def update_experiment(
        self, experiment_id: int, updates: Dict[str, Any]
    ) -> bool:
        """Update specific fields on an experiment."""
        now = _now()
        std_updates = {}
        dyn_updates = {}

        for key, value in updates.items():
            db_col = _db_col(key)
            if db_col in STANDARD_COLUMNS:
                std_updates[db_col] = value
            elif key not in ('experiment_id', 'created_at', 'updated_at'):
                dyn_updates[key] = str(value) if value is not None else ''

        if std_updates:
            set_parts = [f"{col} = ?" for col in std_updates.keys()]
            set_parts.append("updated_at = ?")
            vals = list(std_updates.values()) + [now, experiment_id]
            self._conn.execute(
                f"UPDATE experiments SET {', '.join(set_parts)} WHERE experiment_id = ?",
                vals,
            )

        for key, value in dyn_updates.items():
            self._ensure_column(key)
            self._conn.execute(
                f"UPDATE experiments SET {key} = ?, updated_at = ? WHERE experiment_id = ?",
                (value, now, experiment_id),
            )

        if std_updates or dyn_updates:
            self._update_fts(experiment_id)

        return True

    def delete_experiment(self, experiment_id: int) -> bool:
        """Delete an experiment."""
        try:
            self._conn.execute(
                "DELETE FROM experiments_fts WHERE rowid = ?", (experiment_id,)
            )
        except sqlite3.OperationalError:
            pass

        cursor = self._conn.execute(
            "DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,)
        )
        return cursor.rowcount > 0

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a Row to dict, mapping column names back to dict keys."""
        d = {}
        for key in row.keys():
            val = row[key]
            out_key = _dict_key(key)
            d[out_key] = val
        return d

    def _update_fts(self, experiment_id: int):
        """Update FTS index for an experiment (standalone FTS5 table)."""
        try:
            row = self._conn.execute(
                "SELECT file_name, experiment_name, strain, stim_type, animal_id, tags, notes, protocol "
                "FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            if row:
                # Delete old entry (standalone FTS — just delete by rowid)
                self._conn.execute(
                    "DELETE FROM experiments_fts WHERE rowid = ?", (experiment_id,)
                )
                self._conn.execute(
                    "INSERT INTO experiments_fts(rowid, file_name, experiment_name, strain, stim_type, animal_id, tags, notes, protocol) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (experiment_id, row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]),
                )
        except sqlite3.OperationalError:
            pass  # FTS5 not available

    # ------------------------------------------------------------------
    # Dynamic columns (ALTER TABLE)
    # ------------------------------------------------------------------

    def _ensure_column(self, col_name: str):
        """Add column to experiments table if it doesn't exist."""
        # Check if column exists
        cols = self.get_all_columns()
        if col_name not in cols:
            self.add_column(col_name, 'TEXT')

    def add_column(self, col_name: str, col_type: str = 'TEXT') -> bool:
        """Add a dynamic column to the experiments table."""
        try:
            self._conn.execute(
                f"ALTER TABLE experiments ADD COLUMN {col_name} {col_type} DEFAULT ''"
            )
            return True
        except sqlite3.OperationalError:
            return False  # Column already exists

    def get_all_columns(self) -> Dict[str, str]:
        """Get all columns on experiments table via PRAGMA table_info."""
        rows = self._conn.execute("PRAGMA table_info(experiments)").fetchall()
        return {r["name"]: r["type"] for r in rows}

    def get_dynamic_columns(self) -> List[Dict[str, str]]:
        """Get columns that are NOT in the default schema (user-added)."""
        all_cols = self.get_all_columns()
        dynamic = []
        for name, col_type in all_cols.items():
            if name not in STANDARD_COLUMNS and name not in (
                'experiment_id', 'created_at', 'updated_at'
            ):
                dynamic.append({"column_key": name, "column_type": col_type, "display_name": name})
        return dynamic

    # ------------------------------------------------------------------
    # Sources CRUD
    # ------------------------------------------------------------------

    def add_source(self, file_path: str, file_hash: str = '') -> int:
        """Register a reference document. Returns source_id."""
        p = Path(file_path)
        now = _now()

        existing = self._conn.execute(
            "SELECT source_id FROM sources WHERE file_path = ?", (file_path,)
        ).fetchone()
        if existing:
            # Update hash if changed
            if file_hash:
                self._conn.execute(
                    "UPDATE sources SET file_hash = ? WHERE source_id = ?",
                    (file_hash, existing["source_id"]),
                )
            return existing["source_id"]

        cursor = self._conn.execute(
            "INSERT INTO sources (file_path, file_name, file_type, file_hash, created_at) VALUES (?, ?, ?, ?, ?)",
            (file_path, p.name, p.suffix.lstrip('.').lower(), file_hash, now),
        )
        return cursor.lastrowid

    def get_source(self, source_id: int) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_sources(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM sources ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_source(self, source_id: int, parsed_content: str) -> bool:
        """Update parsed content cache for a source."""
        self._conn.execute(
            "UPDATE sources SET parsed_content = ?, parsed_at = ? WHERE source_id = ?",
            (parsed_content, _now(), source_id),
        )
        return True

    # ------------------------------------------------------------------
    # Source links CRUD
    # ------------------------------------------------------------------

    def add_link(
        self, source_id: int, animal_id: str, field: str, value: str,
        experiment_id: Optional[int] = None,
        location: str = '', confidence: float = 0.5,
    ) -> int:
        """Create a source_link. Returns link_id."""
        cursor = self._conn.execute(
            """INSERT INTO source_links
               (source_id, animal_id, experiment_id, field, value, location, confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (source_id, animal_id, experiment_id, field, value, location, confidence, _now()),
        )
        return cursor.lastrowid

    def get_links(
        self,
        animal_id: Optional[str] = None,
        experiment_id: Optional[int] = None,
        field: Optional[str] = None,
        source_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query source_links with optional filters."""
        where_parts = []
        params: list = []

        if animal_id is not None:
            where_parts.append("sl.animal_id = ?")
            params.append(animal_id)
        if experiment_id is not None:
            where_parts.append("sl.experiment_id = ?")
            params.append(experiment_id)
        if field is not None:
            where_parts.append("sl.field = ?")
            params.append(field)
        if source_id is not None:
            where_parts.append("sl.source_id = ?")
            params.append(source_id)

        where = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        rows = self._conn.execute(
            f"""SELECT sl.*, s.file_path as source_path, s.file_name as source_name
                FROM source_links sl
                JOIN sources s ON s.source_id = sl.source_id
                {where}
                ORDER BY sl.link_id DESC""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_links_for_animal(self, animal_id: str) -> List[Dict[str, Any]]:
        """Get all source links for an animal."""
        return self.get_links(animal_id=animal_id)

    # ------------------------------------------------------------------
    # Snapshots CRUD
    # ------------------------------------------------------------------

    def create_snapshot(
        self, name: str, filter_query: Optional[Dict] = None,
        data_directory: str = '', description: str = '',
    ) -> int:
        """Create or update a snapshot. Returns snapshot_id."""
        now = _now()
        fq = json.dumps(filter_query or {})

        existing = self._conn.execute(
            "SELECT snapshot_id FROM snapshots WHERE name = ?", (name,)
        ).fetchone()

        if existing:
            sid = existing["snapshot_id"]
            self._conn.execute(
                """UPDATE snapshots SET filter_query=?, data_directory=?, description=?, updated_at=?
                   WHERE snapshot_id=?""",
                (fq, data_directory, description, now, sid),
            )
            return sid

        cursor = self._conn.execute(
            """INSERT INTO snapshots (name, description, filter_query, data_directory, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (name, description, fq, data_directory, now, now),
        )
        return cursor.lastrowid

    def get_snapshot(self, snapshot_id: int) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
        ).fetchone()
        if row:
            d = dict(row)
            d['filter_query'] = json.loads(d.get('filter_query', '{}'))
            return d
        return None

    def get_snapshot_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM snapshots WHERE name = ?", (name,)
        ).fetchone()
        if row:
            d = dict(row)
            d['filter_query'] = json.loads(d.get('filter_query', '{}'))
            return d
        return None

    def get_snapshot_by_directory(self, data_directory: str) -> Optional[Dict[str, Any]]:
        """Find snapshot by data directory path."""
        norm_dir = str(Path(data_directory).resolve())
        row = self._conn.execute(
            "SELECT * FROM snapshots WHERE data_directory = ?", (norm_dir,)
        ).fetchone()
        if row:
            d = dict(row)
            d['filter_query'] = json.loads(d.get('filter_query', '{}'))
            return d
        return None

    def list_snapshots(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM snapshots ORDER BY updated_at DESC"
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d['filter_query'] = json.loads(d.get('filter_query', '{}'))
            result.append(d)
        return result

    def update_snapshot_json_sync(
        self, snapshot_id: int, json_path: str, json_mtime: float, json_hash: str
    ):
        """Update JSON sync metadata for a snapshot."""
        self._conn.execute(
            """UPDATE snapshots SET json_path=?, json_mtime=?, json_hash=?, updated_at=?
               WHERE snapshot_id=?""",
            (json_path, json_mtime, json_hash, _now(), snapshot_id),
        )

    # ------------------------------------------------------------------
    # Full-text search
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Full-text search across experiments using FTS5."""
        try:
            rows = self._conn.execute(
                """SELECT e.* FROM experiments_fts fts
                   JOIN experiments e ON e.experiment_id = fts.rowid
                   WHERE experiments_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (query, limit),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except sqlite3.OperationalError:
            # FTS5 not available — fallback to LIKE
            like = f"%{query}%"
            rows = self._conn.execute(
                """SELECT * FROM experiments
                   WHERE file_name LIKE ? OR experiment_name LIKE ? OR strain LIKE ?
                     OR stim_type LIKE ? OR animal_id LIKE ? OR notes LIKE ?
                   LIMIT ?""",
                (like, like, like, like, like, like, limit),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Unique values & completeness
    # ------------------------------------------------------------------

    def get_unique_values(self, field: str) -> List[str]:
        """Get sorted unique non-empty values for a field."""
        col = _db_col(field)
        # Check column exists
        all_cols = self.get_all_columns()
        if col not in all_cols:
            return []

        rows = self._conn.execute(
            f"""SELECT DISTINCT {col} FROM experiments
                WHERE {col} != '' AND {col} IS NOT NULL
                  AND {col} NOT IN ('Loading...', 'Unknown')
                ORDER BY {col}""",
        ).fetchall()
        return [r[0] for r in rows]

    def get_metadata_completeness(self) -> Dict[str, Any]:
        """Get fill percentage for metadata columns."""
        total = self._conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        if total == 0:
            return {"per_column": {}, "overall_pct": 0.0, "total_experiments": 0}

        fields = ["experiment_name", "strain", "stim_type", "power", "sex",
                   "animal_id", "channel", "stim_channel", "protocol"]

        per_column = {}
        total_filled = 0
        total_cells = 0

        for col in fields:
            filled = self._conn.execute(
                f"""SELECT COUNT(*) FROM experiments
                    WHERE {col} != '' AND {col} IS NOT NULL
                      AND {col} NOT IN ('Loading...', 'Unknown', 'pending')""",
            ).fetchone()[0]
            pct = (filled / total) * 100
            display_key = _dict_key(col)
            per_column[display_key] = {
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
            "total_experiments": total,
        }

    def get_experiment_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]

    # ------------------------------------------------------------------
    # JSON export / import (for .physiometrics snapshot files)
    # ------------------------------------------------------------------

    def export_snapshot(self, snapshot_id: int) -> Dict[str, Any]:
        """Export a snapshot's experiments to JSON for .physiometrics file."""
        snapshot = self.get_snapshot(snapshot_id)
        if not snapshot:
            raise ValueError(f"Snapshot {snapshot_id} not found")

        # Get experiments matching snapshot filter
        fq = snapshot.get('filter_query', {})
        experiments, _ = self.get_experiments(**fq)

        # Convert to JSON-friendly format
        files_json = []
        for exp in experiments:
            entry = dict(exp)
            entry.pop('experiment_id', None)
            entry.pop('created_at', None)
            entry.pop('updated_at', None)
            files_json.append(entry)

        return {
            "version": 3,
            "snapshot_name": snapshot["name"],
            "data_directory": snapshot.get("data_directory", ""),
            "description": snapshot.get("description", ""),
            "filter_query": fq,
            "created": snapshot["created_at"],
            "last_modified": _now(),
            "experiment_count": len(files_json),
            "experiments": files_json,
            "custom_columns": self.get_dynamic_columns(),
        }

    def import_from_json(self, json_data: Dict[str, Any], data_directory: str = '') -> Dict[str, Any]:
        """Import experiments from a .physiometrics JSON file.

        Handles both v2 (files-based) and v3 (experiments-based) formats.
        Returns merge report: {new, updated, unchanged}.
        """
        now = _now()
        new_count = 0
        updated_count = 0
        unchanged_count = 0

        # Support both v2 "files" and v3 "experiments" keys
        entries = json_data.get("experiments", json_data.get("files", []))

        for entry in entries:
            # Skip flat subrow entries from v2 format
            if entry.get("is_sub_row"):
                continue

            file_path = entry.get("file_path", "")
            if not file_path:
                continue

            # Resolve relative paths
            if data_directory and not Path(file_path).is_absolute():
                file_path = str(Path(data_directory) / file_path)

            # Map legacy key names
            mapped = {}
            for k, v in entry.items():
                mapped[k] = v
            mapped['file_path'] = file_path

            # Check existing
            channel = mapped.get('channel', '')
            animal_id = mapped.get('animal_id', '')
            existing = self.get_experiment_by_key(file_path, channel, animal_id)

            if existing is None:
                self.upsert_experiment(mapped)
                new_count += 1
            else:
                # Check if any fields differ
                changed = False
                updates = {}
                for key, val in mapped.items():
                    if key in ('file_path', 'file_name', 'exports', 'subrows',
                               'path_keywords', 'stim_channels', 'stim_frequency',
                               'is_sub_row', 'parent_file', 'subrow_count',
                               'file_id', 'project_id', 'updated_at', 'field_timestamps',
                               'linked_notes'):
                        continue
                    val_str = str(val) if val is not None else ''
                    if not val_str or val_str in ('', 'Loading...', 'Unknown'):
                        continue
                    existing_val = str(existing.get(_dict_key(key), ''))
                    if not existing_val and val_str:
                        updates[key] = val_str
                        changed = True

                if changed and updates:
                    self.update_experiment(existing['experiment_id'], updates)
                    updated_count += 1
                else:
                    unchanged_count += 1

        # Handle v2 subrows: create separate experiment entries
        for entry in entries:
            if entry.get("is_sub_row"):
                continue
            subrows = entry.get("subrows", [])
            if not subrows:
                continue

            parent_path = entry.get("file_path", "")
            if data_directory and not Path(parent_path).is_absolute():
                parent_path = str(Path(data_directory) / parent_path)

            for sr in subrows:
                sr_data = dict(entry)
                sr_data.pop('subrows', None)
                sr_data['file_path'] = parent_path
                # Override with subrow-specific fields
                for sk in ('channel', 'animal_id', 'sex', 'group', 'group_name',
                           'protocol', 'stim_type', 'power', 'experiment',
                           'experiment_name', 'strain'):
                    if sk in sr and sr[sk]:
                        sr_data[sk] = sr[sk]

                existing = self.get_experiment_by_key(
                    parent_path,
                    sr_data.get('channel', ''),
                    sr_data.get('animal_id', ''),
                )
                if existing is None:
                    self.upsert_experiment(sr_data)
                    new_count += 1

        return {"new": new_count, "updated": updated_count, "unchanged": unchanged_count}

    # ------------------------------------------------------------------
    # Grouped view
    # ------------------------------------------------------------------

    def get_experiments_grouped(self) -> Dict[str, Any]:
        """Return experiments grouped by file_path with summary stats."""
        experiments, total = self.get_experiments()

        if not experiments:
            return {"total_experiments": 0, "files": []}

        file_map: Dict[str, List[Dict]] = {}
        for exp in experiments:
            fp = exp.get("file_path", "")
            file_map.setdefault(fp, []).append(exp)

        fields_to_check = ["experiment", "strain", "stim_type", "power", "sex",
                           "animal_id", "channel", "protocol"]

        files = []
        for file_path in sorted(file_map.keys()):
            exps = file_map[file_path]
            count = len(exps)

            total_cells = count * len(fields_to_check)
            filled_cells = 0
            for exp in exps:
                for field in fields_to_check:
                    val = exp.get(field, "")
                    if val and str(val).strip() and str(val) not in ("Loading...", "Unknown", "pending"):
                        filled_cells += 1
            filled_pct = round((filled_cells / total_cells * 100) if total_cells else 0, 1)

            common = {}
            for field in ["strain", "animal_id", "stim_type", "power", "experiment"]:
                vals = set()
                for exp in exps:
                    v = exp.get(field, "")
                    if v and str(v).strip():
                        vals.add(str(v).strip())
                if len(vals) == 1:
                    common[field] = vals.pop()

            entry = {
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "experiment_count": count,
                "filled_pct": filled_pct,
            }
            entry.update(common)
            if count > 1:
                entry["animals"] = [
                    {"animal_id": e.get("animal_id", ""), "channel": e.get("channel", "")}
                    for e in exps
                ]
            files.append(entry)

        return {"total_experiments": total, "file_count": len(files), "files": files}

    # ------------------------------------------------------------------
    # Backup & restore
    # ------------------------------------------------------------------

    def backup(self, trigger_event: str = "manual") -> str:
        """Create a backup using SQLite's backup API. Returns backup path."""
        backup_path = _backup_dir() / f"PhysioMetrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"

        backup_conn = sqlite3.connect(str(backup_path))
        try:
            self._conn.backup(backup_conn)
        finally:
            backup_conn.close()

        self._conn.execute(
            "INSERT INTO backups (backup_path, trigger_event, created_at) VALUES (?, ?, ?)",
            (str(backup_path), trigger_event, _now()),
        )

        self._prune_backups()
        return str(backup_path)

    def _prune_backups(self, keep: int = 10):
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
        rows = self._conn.execute(
            "SELECT * FROM backups ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

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

    def close(self):
        if self._conn:
            self._conn.close()
