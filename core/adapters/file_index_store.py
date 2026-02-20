"""
File Index SQLite adapter — local cache for lab file inventory and notes text.

DB location: %APPDATA%/PhysioMetrics/file_index.db
Uses WAL mode, FTS5, CASCADE deletes.
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from contextlib import contextmanager


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS files (
    file_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path   TEXT NOT NULL UNIQUE,
    file_name   TEXT NOT NULL,
    parent_dir  TEXT NOT NULL,
    extension   TEXT NOT NULL,
    file_size   INTEGER NOT NULL,
    mtime       TEXT NOT NULL,
    file_class  TEXT NOT NULL,
    indexed_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_files_class ON files(file_class);
CREATE INDEX IF NOT EXISTS idx_files_ext ON files(extension);
CREATE INDEX IF NOT EXISTS idx_files_parent ON files(parent_dir);

CREATE TABLE IF NOT EXISTS notes_cells (
    cell_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id     INTEGER NOT NULL REFERENCES files ON DELETE CASCADE,
    sheet_name  TEXT NOT NULL DEFAULT '',
    row_num     INTEGER NOT NULL,
    col_num     INTEGER NOT NULL,
    value       TEXT NOT NULL,
    indexed_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cells_file ON notes_cells(file_id);
CREATE INDEX IF NOT EXISTS idx_cells_sheet ON notes_cells(file_id, sheet_name);

CREATE TABLE IF NOT EXISTS index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    value,
    content='notes_cells',
    content_rowid='cell_id',
    tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
    file_name, parent_dir, file_path,
    content='files',
    content_rowid='file_id',
    tokenize='porter unicode61'
);
"""

# Triggers to keep FTS in sync
FTS_TRIGGERS_SQL = """
CREATE TRIGGER IF NOT EXISTS notes_cells_ai AFTER INSERT ON notes_cells BEGIN
    INSERT INTO notes_fts(rowid, value) VALUES (new.cell_id, new.value);
END;
CREATE TRIGGER IF NOT EXISTS notes_cells_ad AFTER DELETE ON notes_cells BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, value) VALUES('delete', old.cell_id, old.value);
END;
CREATE TRIGGER IF NOT EXISTS notes_cells_au AFTER UPDATE ON notes_cells BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, value) VALUES('delete', old.cell_id, old.value);
    INSERT INTO notes_fts(rowid, value) VALUES (new.cell_id, new.value);
END;

CREATE TRIGGER IF NOT EXISTS files_ai AFTER INSERT ON files BEGIN
    INSERT INTO files_fts(rowid, file_name, parent_dir, file_path)
    VALUES (new.file_id, new.file_name, new.parent_dir, new.file_path);
END;
CREATE TRIGGER IF NOT EXISTS files_ad AFTER DELETE ON files BEGIN
    INSERT INTO files_fts(files_fts, rowid, file_name, parent_dir, file_path)
    VALUES('delete', old.file_id, old.file_name, old.parent_dir, old.file_path);
END;
CREATE TRIGGER IF NOT EXISTS files_au AFTER UPDATE ON files BEGIN
    INSERT INTO files_fts(files_fts, rowid, file_name, parent_dir, file_path)
    VALUES('delete', old.file_id, old.file_name, old.parent_dir, old.file_path);
    INSERT INTO files_fts(rowid, file_name, parent_dir, file_path)
    VALUES (new.file_id, new.file_name, new.parent_dir, new.file_path);
END;
"""


def _default_db_path() -> Path:
    appdata = os.environ.get("APPDATA", str(Path.home()))
    return Path(appdata) / "PhysioMetrics" / "file_index.db"


class FileIndexStore:
    """SQLite store for the lab file index."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=10)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        cur = self._conn.cursor()
        cur.executescript(SCHEMA_SQL)
        # FTS tables need separate creation (can't be in executescript with IF NOT EXISTS reliably)
        try:
            cur.executescript(FTS_SQL)
        except sqlite3.OperationalError:
            pass  # Already exists
        try:
            cur.executescript(FTS_TRIGGERS_SQL)
        except sqlite3.OperationalError:
            pass  # Already exists
        self._conn.commit()

    @contextmanager
    def _transaction(self):
        cur = self._conn.cursor()
        cur.execute("BEGIN")
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # --- File operations ---

    def upsert_file(self, file_path: str, file_name: str, parent_dir: str,
                    extension: str, file_size: int, mtime: str,
                    file_class: str) -> int:
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute("""
            INSERT INTO files (file_path, file_name, parent_dir, extension, file_size, mtime, file_class, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                file_name=excluded.file_name, parent_dir=excluded.parent_dir,
                extension=excluded.extension, file_size=excluded.file_size,
                mtime=excluded.mtime, file_class=excluded.file_class,
                indexed_at=excluded.indexed_at
        """, (file_path, file_name, parent_dir, extension, file_size, mtime, file_class, now))
        self._conn.commit()
        # Get the file_id
        row = self._conn.execute("SELECT file_id FROM files WHERE file_path=?", (file_path,)).fetchone()
        return row["file_id"]

    def get_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute("SELECT * FROM files WHERE file_path=?", (file_path,)).fetchone()
        return dict(row) if row else None

    def get_file_by_id(self, file_id: int) -> Optional[Dict[str, Any]]:
        row = self._conn.execute("SELECT * FROM files WHERE file_id=?", (file_id,)).fetchone()
        return dict(row) if row else None

    def get_files(self, file_class: Optional[str] = None, extension: Optional[str] = None,
                  path_contains: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM files WHERE 1=1"
        params = []
        if file_class:
            sql += " AND file_class=?"
            params.append(file_class)
        if extension:
            sql += " AND extension=?"
            params.append(extension)
        if path_contains:
            sql += " AND file_path LIKE ?"
            params.append(f"%{path_contains}%")
        sql += " ORDER BY file_path LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def delete_file(self, file_path: str):
        self._conn.execute("DELETE FROM files WHERE file_path=?", (file_path,))
        self._conn.commit()

    def delete_missing_files(self, existing_paths: set) -> int:
        """Delete files from index that no longer exist on disk. Returns count deleted."""
        all_rows = self._conn.execute("SELECT file_id, file_path FROM files").fetchall()
        to_delete = [r["file_id"] for r in all_rows if r["file_path"] not in existing_paths]
        if to_delete:
            placeholders = ",".join("?" * len(to_delete))
            self._conn.execute(f"DELETE FROM files WHERE file_id IN ({placeholders})", to_delete)
            self._conn.commit()
        return len(to_delete)

    # --- Notes cell operations ---

    def upsert_cells(self, file_id: int, cells: List[Dict[str, Any]]):
        """Replace all cells for a file. cells: [{sheet_name, row_num, col_num, value}]"""
        now = datetime.now(timezone.utc).isoformat()
        with self._transaction() as cur:
            # Delete old cells (CASCADE triggers FTS cleanup via triggers)
            cur.execute("DELETE FROM notes_cells WHERE file_id=?", (file_id,))
            # Bulk insert
            cur.executemany("""
                INSERT INTO notes_cells (file_id, sheet_name, row_num, col_num, value, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [(file_id, c.get("sheet_name", ""), c["row_num"], c["col_num"], c["value"], now)
                  for c in cells])

    def get_cells(self, file_id: int, sheet_name: Optional[str] = None,
                  limit: int = 5000) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM notes_cells WHERE file_id=?"
        params: list = [file_id]
        if sheet_name is not None:
            sql += " AND sheet_name=?"
            params.append(sheet_name)
        sql += " ORDER BY sheet_name, row_num, col_num LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_sheets(self, file_id: int) -> List[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT sheet_name FROM notes_cells WHERE file_id=? ORDER BY sheet_name",
            (file_id,)
        ).fetchall()
        return [r["sheet_name"] for r in rows]

    def has_cells(self, file_id: int) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM notes_cells WHERE file_id=? LIMIT 1", (file_id,)
        ).fetchone()
        return row is not None

    # --- FTS5 search ---

    def search_files(self, query: str, file_class: Optional[str] = None,
                     limit: int = 50) -> List[Dict[str, Any]]:
        """FTS5 search on file names/paths."""
        fts_query = self._sanitize_fts(query)
        if not fts_query:
            return []
        sql = """
            SELECT f.*, rank
            FROM files_fts fts
            JOIN files f ON f.file_id = fts.rowid
            WHERE files_fts MATCH ?
        """
        params: list = [fts_query]
        if file_class:
            sql += " AND f.file_class=?"
            params.append(file_class)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def search_notes(self, query: str, file_id: Optional[int] = None,
                     limit: int = 100) -> List[Dict[str, Any]]:
        """FTS5 search on notes cell text. Returns cells with file context."""
        fts_query = self._sanitize_fts(query)
        if not fts_query:
            return []
        sql = """
            SELECT nc.*, f.file_path, f.file_name,
                   snippet(notes_fts, 0, '>>>', '<<<', '...', 30) as snippet,
                   rank
            FROM notes_fts fts
            JOIN notes_cells nc ON nc.cell_id = fts.rowid
            JOIN files f ON f.file_id = nc.file_id
            WHERE notes_fts MATCH ?
        """
        params: list = [fts_query]
        if file_id:
            sql += " AND nc.file_id=?"
            params.append(file_id)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def find_value_in_notes(self, value: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Exact or prefix match on cell values (for file name lookups)."""
        sql = """
            SELECT nc.*, f.file_path, f.file_name
            FROM notes_cells nc
            JOIN files f ON f.file_id = nc.file_id
            WHERE nc.value LIKE ?
            ORDER BY f.file_path, nc.sheet_name, nc.row_num
            LIMIT ?
        """
        rows = self._conn.execute(sql, (f"%{value}%", limit)).fetchall()
        return [dict(r) for r in rows]

    # --- Metadata ---

    def set_meta(self, key: str, value: str):
        self._conn.execute(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
            (key, value)
        )
        self._conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        row = self._conn.execute("SELECT value FROM index_meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    # --- Stats ---

    def get_stats(self) -> Dict[str, Any]:
        stats = {}
        # File counts by class
        rows = self._conn.execute(
            "SELECT file_class, COUNT(*) as cnt FROM files GROUP BY file_class ORDER BY file_class"
        ).fetchall()
        stats["files_by_class"] = {r["file_class"]: r["cnt"] for r in rows}
        stats["total_files"] = sum(stats["files_by_class"].values())

        # Cell count
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM notes_cells").fetchone()
        stats["total_cells"] = row["cnt"]

        # Files with cells extracted
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT file_id) as cnt FROM notes_cells"
        ).fetchone()
        stats["files_extracted"] = row["cnt"]

        # Extractable files (notes + reference)
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM files WHERE file_class IN ('notes', 'reference')"
        ).fetchone()
        stats["files_extractable"] = row["cnt"]

        # Scan roots
        roots = self.get_meta("scan_roots")
        stats["scan_roots"] = roots if roots else ""

        # Last scan/extract times
        stats["last_scan"] = self.get_meta("last_scan") or "never"
        stats["last_extract"] = self.get_meta("last_extract") or "never"

        return stats

    @staticmethod
    def _sanitize_fts(query: str) -> str:
        """Sanitize FTS5 query — escape special chars, handle simple terms."""
        if not query or not query.strip():
            return ""
        # If it looks like a raw FTS query (has operators), pass through
        if any(op in query for op in [" AND ", " OR ", " NOT ", '"', "*"]):
            return query
        # Otherwise, quote each term for exact prefix matching
        terms = query.strip().split()
        return " ".join(f'"{t}"' for t in terms if t)
