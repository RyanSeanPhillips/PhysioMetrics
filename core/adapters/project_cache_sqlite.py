"""
SQLite cache for project metadata extraction.

Stores parsed notes, extraction patterns, field vocabulary, and
key-value knowledge. WAL mode, same conventions as code_index_sqlite.py.

DB location: <project_data_dir>/.physiometrics_cache.db
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS pj_notes_parsed (
    note_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path   TEXT NOT NULL,
    file_hash   TEXT NOT NULL UNIQUE,
    content     TEXT NOT NULL,
    abf_refs    TEXT DEFAULT '[]',
    parsed_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pj_notes_path ON pj_notes_parsed(file_path);

CREATE TABLE IF NOT EXISTS pj_extraction_patterns (
    pattern_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    type        TEXT NOT NULL,
    source      TEXT NOT NULL,
    target      TEXT NOT NULL,
    extractor   TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 0.5,
    usage_count INTEGER NOT NULL DEFAULT 0,
    notes       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pj_patterns_target ON pj_extraction_patterns(target);
CREATE INDEX IF NOT EXISTS idx_pj_patterns_type ON pj_extraction_patterns(type);

CREATE TABLE IF NOT EXISTS pj_field_vocabulary (
    vocab_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    field       TEXT NOT NULL,
    value       TEXT NOT NULL,
    usage_count INTEGER NOT NULL DEFAULT 1,
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL,
    UNIQUE(field, value)
);

CREATE INDEX IF NOT EXISTS idx_pj_vocab_field ON pj_field_vocabulary(field);

CREATE TABLE IF NOT EXISTS pj_knowledge (
    key         TEXT PRIMARY KEY,
    value_json  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pj_provenance (
    prov_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path      TEXT NOT NULL,
    field          TEXT NOT NULL,
    value          TEXT NOT NULL,
    source_type    TEXT NOT NULL,
    source_detail  TEXT DEFAULT '',
    source_preview TEXT DEFAULT '',
    confidence     REAL DEFAULT 0.5,
    reason         TEXT DEFAULT '',
    supersedes     INTEGER,
    created_at     TEXT NOT NULL,
    FOREIGN KEY (supersedes) REFERENCES pj_provenance(prov_id)
);

CREATE INDEX IF NOT EXISTS idx_pj_prov_file ON pj_provenance(file_path);
CREATE INDEX IF NOT EXISTS idx_pj_prov_field ON pj_provenance(file_path, field);

CREATE TABLE IF NOT EXISTS pj_file_labels (
    label_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path   TEXT NOT NULL,
    label       TEXT NOT NULL,
    features    TEXT DEFAULT '{}',
    created_at  TEXT NOT NULL,
    UNIQUE(file_path)
);
"""


class ProjectCacheSQLite:
    """SQLite cache for project metadata extraction patterns and parsed notes."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript(SCHEMA_SQL)

    @contextmanager
    def _transaction(self):
        self._conn.execute("BEGIN")
        try:
            yield
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def _now(self) -> str:
        return datetime.now().isoformat()

    # ------------------------------------------------------------------
    # Notes cache
    # ------------------------------------------------------------------

    def save_notes_cache(
        self, path: str, file_hash: str, content: Any, abf_refs: List[str]
    ) -> None:
        """Cache parsed notes content keyed by file hash."""
        self._conn.execute(
            """INSERT INTO pj_notes_parsed (file_path, file_hash, content, abf_refs, parsed_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(file_hash) DO UPDATE SET
                 file_path=excluded.file_path,
                 content=excluded.content,
                 abf_refs=excluded.abf_refs,
                 parsed_at=excluded.parsed_at""",
            (path, file_hash, json.dumps(content, default=str), json.dumps(abf_refs), self._now()),
        )

    def get_notes_cache(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached notes by file hash. Returns None if not cached."""
        row = self._conn.execute(
            "SELECT * FROM pj_notes_parsed WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        if row:
            return {
                "file_path": row["file_path"],
                "file_hash": row["file_hash"],
                "content": json.loads(row["content"]),
                "abf_refs": json.loads(row["abf_refs"]),
                "parsed_at": row["parsed_at"],
            }
        return None

    def invalidate_notes_cache(self, file_hash: str) -> bool:
        """Remove a cached notes entry by hash."""
        cursor = self._conn.execute(
            "DELETE FROM pj_notes_parsed WHERE file_hash = ?", (file_hash,)
        )
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Extraction patterns
    # ------------------------------------------------------------------

    def save_pattern(
        self,
        type: str,
        source: str,
        target: str,
        extractor: str,
        confidence: float = 0.5,
        notes: Optional[str] = None,
    ) -> int:
        """
        Store an extraction pattern.

        Args:
            type: Pattern type (e.g. 'subfolder', 'filename', 'notes_column', 'regex').
            source: What to match (e.g. subfolder name, column header, regex).
            target: Target metadata field (e.g. 'strain', 'stim_type').
            extractor: How to extract value (e.g. 'literal:VgatCre', 'regex:\\d+mW').
            confidence: 0.0-1.0 confidence score.
            notes: Optional human-readable explanation.

        Returns:
            Pattern ID.
        """
        now = self._now()
        cursor = self._conn.execute(
            """INSERT INTO pj_extraction_patterns
               (type, source, target, extractor, confidence, notes, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (type, source, target, extractor, confidence, notes, now, now),
        )
        return cursor.lastrowid

    def get_patterns(
        self,
        target_field: Optional[str] = None,
        type: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve extraction patterns, sorted by confidence * usage_count descending.

        Args:
            target_field: Filter by target metadata field.
            type: Filter by pattern type.
            min_confidence: Minimum confidence threshold.
        """
        sql = "SELECT * FROM pj_extraction_patterns WHERE confidence >= ?"
        params: list = [min_confidence]

        if target_field:
            sql += " AND target = ?"
            params.append(target_field)
        if type:
            sql += " AND type = ?"
            params.append(type)

        sql += " ORDER BY (confidence * (usage_count + 1)) DESC"

        rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "pattern_id": row["pattern_id"],
                "type": row["type"],
                "source": row["source"],
                "target": row["target"],
                "extractor": row["extractor"],
                "confidence": row["confidence"],
                "usage_count": row["usage_count"],
                "notes": row["notes"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def increment_pattern_usage(self, pattern_id: int) -> bool:
        """Increment usage count for a pattern (called when it's successfully applied)."""
        cursor = self._conn.execute(
            """UPDATE pj_extraction_patterns
               SET usage_count = usage_count + 1, updated_at = ?
               WHERE pattern_id = ?""",
            (self._now(), pattern_id),
        )
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Field vocabulary
    # ------------------------------------------------------------------

    def add_vocabulary(self, field: str, value: str) -> None:
        """Record a known value for a metadata field. UPSERT — increments usage_count."""
        now = self._now()
        self._conn.execute(
            """INSERT INTO pj_field_vocabulary (field, value, usage_count, first_seen, last_seen)
               VALUES (?, ?, 1, ?, ?)
               ON CONFLICT(field, value) DO UPDATE SET
                 usage_count = usage_count + 1,
                 last_seen = excluded.last_seen""",
            (field, value, now, now),
        )

    def get_vocabulary(self, field: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get known values for a field (or all fields), sorted by usage_count descending.
        """
        if field:
            rows = self._conn.execute(
                "SELECT * FROM pj_field_vocabulary WHERE field = ? ORDER BY usage_count DESC",
                (field,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM pj_field_vocabulary ORDER BY field, usage_count DESC"
            ).fetchall()

        return [
            {
                "field": row["field"],
                "value": row["value"],
                "usage_count": row["usage_count"],
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Knowledge cache (key-value)
    # ------------------------------------------------------------------

    def set_knowledge(self, key: str, value: Any) -> None:
        """Store a key-value pair."""
        self._conn.execute(
            """INSERT INTO pj_knowledge (key, value_json, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at""",
            (key, json.dumps(value, default=str), self._now()),
        )

    def get_knowledge(self, key: str) -> Optional[Any]:
        """Retrieve a value by key. Returns the deserialized value or None."""
        row = self._conn.execute(
            "SELECT value_json FROM pj_knowledge WHERE key = ?", (key,)
        ).fetchone()
        if row:
            return json.loads(row["value_json"])
        return None

    # ------------------------------------------------------------------
    # Provenance tracking
    # ------------------------------------------------------------------

    def add_provenance(
        self,
        file_path: str,
        field: str,
        value: str,
        source_type: str,
        source_detail: str = "",
        source_preview: str = "",
        confidence: float = 0.5,
        reason: str = "",
    ) -> int:
        """
        Record provenance for a metadata field value.

        If a previous provenance record exists for the same file+field,
        the new record's `supersedes` points to it (correction chain).

        Returns:
            prov_id of the new record.
        """
        # Find previous record for this file+field to build correction chain
        prev = self._conn.execute(
            """SELECT prov_id FROM pj_provenance
               WHERE file_path = ? AND field = ?
               ORDER BY prov_id DESC LIMIT 1""",
            (file_path, field),
        ).fetchone()
        supersedes = prev["prov_id"] if prev else None

        cursor = self._conn.execute(
            """INSERT INTO pj_provenance
               (file_path, field, value, source_type, source_detail, source_preview,
                confidence, reason, supersedes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (file_path, field, value, source_type, source_detail, source_preview,
             confidence, reason, supersedes, self._now()),
        )
        return cursor.lastrowid

    def get_provenance(
        self, file_path: str, field: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get provenance records for a file, optionally filtered by field.

        Returns most recent records first.
        """
        if field:
            rows = self._conn.execute(
                """SELECT * FROM pj_provenance
                   WHERE file_path = ? AND field = ?
                   ORDER BY prov_id DESC""",
                (file_path, field),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM pj_provenance
                   WHERE file_path = ?
                   ORDER BY prov_id DESC""",
                (file_path,),
            ).fetchall()

        return [
            {
                "prov_id": row["prov_id"],
                "file_path": row["file_path"],
                "field": row["field"],
                "value": row["value"],
                "source_type": row["source_type"],
                "source_detail": row["source_detail"],
                "source_preview": row["source_preview"],
                "confidence": row["confidence"],
                "reason": row["reason"],
                "supersedes": row["supersedes"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # File labels (for ML classifier training data)
    # ------------------------------------------------------------------

    def save_file_label(self, file_path: str, label: str, features: Optional[Dict] = None) -> int:
        """Store a file type label (training data for ML classifier)."""
        cursor = self._conn.execute(
            """INSERT INTO pj_file_labels (file_path, label, features, created_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(file_path) DO UPDATE SET
                 label=excluded.label, features=excluded.features, created_at=excluded.created_at""",
            (file_path, label, json.dumps(features or {}), self._now()),
        )
        return cursor.lastrowid

    def get_file_labels(self, label: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get file labels, optionally filtered by label type."""
        if label:
            rows = self._conn.execute(
                "SELECT * FROM pj_file_labels WHERE label = ? ORDER BY created_at DESC",
                (label,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM pj_file_labels ORDER BY created_at DESC"
            ).fetchall()

        return [
            {
                "file_path": row["file_path"],
                "label": row["label"],
                "features": json.loads(row["features"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """Get counts for each cache table."""
        def _count(table: str) -> int:
            return self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        return {
            "cached_notes": _count("pj_notes_parsed"),
            "patterns": _count("pj_extraction_patterns"),
            "vocabulary_entries": _count("pj_field_vocabulary"),
            "knowledge_entries": _count("pj_knowledge"),
            "provenance_records": _count("pj_provenance"),
            "file_labels": _count("pj_file_labels"),
        }

    def close(self) -> None:
        self._conn.close()
