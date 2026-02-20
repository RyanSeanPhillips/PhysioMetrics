"""
SQLite implementation of the code index port.

WAL mode, FTS5, batch inserts, CASCADE deletes.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

from core.ports.code_index_port import CodeIndexPort
from core.domain.code_index.models import (
    FileDef, ImportDef, ClassDef, FunctionDef, AttributeAccess,
    SignalDef, ConnectionDef, CallDef, UIWidget, Diagnostic,
    KnowledgeEntry, IndexStats,
)


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS ci_files (
    file_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    rel_path    TEXT NOT NULL UNIQUE,
    abs_path    TEXT NOT NULL,
    file_hash   TEXT NOT NULL,
    mtime       TEXT NOT NULL,
    line_count  INTEGER NOT NULL DEFAULT 0,
    parse_error TEXT,
    indexed_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ci_imports (
    import_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id     INTEGER NOT NULL REFERENCES ci_files ON DELETE CASCADE,
    module      TEXT NOT NULL,
    name        TEXT,
    alias       TEXT,
    is_from     INTEGER NOT NULL DEFAULT 0,
    line_no     INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ci_imports_module ON ci_imports(module);
CREATE INDEX IF NOT EXISTS idx_ci_imports_name ON ci_imports(name);

CREATE TABLE IF NOT EXISTS ci_classes (
    class_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id     INTEGER NOT NULL REFERENCES ci_files ON DELETE CASCADE,
    name        TEXT NOT NULL,
    bases_json  TEXT DEFAULT '[]',
    decorators_json TEXT DEFAULT '[]',
    docstring   TEXT,
    line_start  INTEGER NOT NULL,
    line_end    INTEGER NOT NULL,
    is_dataclass INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_ci_classes_name ON ci_classes(name);

CREATE TABLE IF NOT EXISTS ci_functions (
    func_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id     INTEGER NOT NULL REFERENCES ci_files ON DELETE CASCADE,
    class_id    INTEGER REFERENCES ci_classes ON DELETE CASCADE,
    name        TEXT NOT NULL,
    params_json TEXT DEFAULT '[]',
    return_type TEXT,
    decorators_json TEXT DEFAULT '[]',
    docstring   TEXT,
    line_start  INTEGER NOT NULL,
    line_end    INTEGER NOT NULL,
    is_property INTEGER NOT NULL DEFAULT 0,
    is_static   INTEGER NOT NULL DEFAULT 0,
    complexity  INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_ci_functions_name ON ci_functions(name);
CREATE INDEX IF NOT EXISTS idx_ci_functions_class ON ci_functions(class_id);

CREATE TABLE IF NOT EXISTS ci_attribute_access (
    access_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id     INTEGER NOT NULL REFERENCES ci_files ON DELETE CASCADE,
    func_id     INTEGER REFERENCES ci_functions ON DELETE CASCADE,
    target      TEXT NOT NULL,
    attr_name   TEXT NOT NULL,
    access_type TEXT NOT NULL DEFAULT 'read',
    line_no     INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ci_attr_target ON ci_attribute_access(target, attr_name);

CREATE TABLE IF NOT EXISTS ci_signals (
    signal_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id     INTEGER NOT NULL REFERENCES ci_files ON DELETE CASCADE,
    class_id    INTEGER REFERENCES ci_classes ON DELETE CASCADE,
    name        TEXT NOT NULL,
    param_types_json TEXT DEFAULT '[]',
    line_no     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS ci_connections (
    conn_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id     INTEGER NOT NULL REFERENCES ci_files ON DELETE CASCADE,
    signal_expr TEXT NOT NULL,
    slot_expr   TEXT NOT NULL,
    line_no     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS ci_calls (
    call_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id     INTEGER NOT NULL REFERENCES ci_files ON DELETE CASCADE,
    func_id     INTEGER REFERENCES ci_functions ON DELETE CASCADE,
    callee_expr TEXT NOT NULL,
    line_no     INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ci_calls_callee ON ci_calls(callee_expr);

CREATE TABLE IF NOT EXISTS ci_ui_widgets (
    widget_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    ui_file     TEXT NOT NULL,
    widget_name TEXT NOT NULL,
    widget_class TEXT NOT NULL,
    parent_name TEXT
);

CREATE INDEX IF NOT EXISTS idx_ci_ui_widgets_name ON ci_ui_widgets(widget_name);

CREATE TABLE IF NOT EXISTS ci_diagnostics (
    diag_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id     INTEGER NOT NULL REFERENCES ci_files ON DELETE CASCADE,
    rule_id     TEXT NOT NULL,
    severity    TEXT NOT NULL DEFAULT 'warning',
    message     TEXT NOT NULL,
    line_no     INTEGER,
    context     TEXT,
    is_resolved INTEGER NOT NULL DEFAULT 0,
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ci_diag_rule ON ci_diagnostics(rule_id, severity);

CREATE TABLE IF NOT EXISTS ci_knowledge (
    key         TEXT PRIMARY KEY,
    value_json  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS ci_fts USING fts5(
    rel_path, class_names, function_names, docstrings,
    tokenize='porter unicode61'
);
"""


class CodeIndexSQLite(CodeIndexPort):
    """SQLite implementation of the code index."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode
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

    # === File operations ===

    def upsert_file(self, f: FileDef) -> FileDef:
        self._conn.execute(
            """INSERT INTO ci_files (rel_path, abs_path, file_hash, mtime, line_count, parse_error, indexed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(rel_path) DO UPDATE SET
                 abs_path=excluded.abs_path,
                 file_hash=excluded.file_hash,
                 mtime=excluded.mtime,
                 line_count=excluded.line_count,
                 parse_error=excluded.parse_error,
                 indexed_at=excluded.indexed_at""",
            (f.rel_path, f.abs_path, f.file_hash, f.mtime, f.line_count, f.parse_error, f.indexed_at)
        )
        row = self._conn.execute(
            "SELECT file_id FROM ci_files WHERE rel_path = ?", (f.rel_path,)
        ).fetchone()
        f.file_id = row["file_id"]
        return f

    def get_file(self, file_id: int) -> Optional[FileDef]:
        row = self._conn.execute(
            "SELECT * FROM ci_files WHERE file_id = ?", (file_id,)
        ).fetchone()
        return self._row_to_file(row) if row else None

    def get_file_by_path(self, rel_path: str) -> Optional[FileDef]:
        row = self._conn.execute(
            "SELECT * FROM ci_files WHERE rel_path = ?", (rel_path,)
        ).fetchone()
        return self._row_to_file(row) if row else None

    def list_files(self) -> List[FileDef]:
        rows = self._conn.execute("SELECT * FROM ci_files ORDER BY rel_path").fetchall()
        return [self._row_to_file(row) for row in rows]

    def delete_file(self, file_id: int) -> bool:
        cursor = self._conn.execute("DELETE FROM ci_files WHERE file_id = ?", (file_id,))
        return cursor.rowcount > 0

    def _row_to_file(self, row) -> FileDef:
        return FileDef(
            file_id=row["file_id"],
            rel_path=row["rel_path"],
            abs_path=row["abs_path"],
            file_hash=row["file_hash"],
            mtime=row["mtime"],
            line_count=row["line_count"],
            parse_error=row["parse_error"],
            indexed_at=row["indexed_at"],
        )

    # === Bulk inserts ===

    def bulk_insert_imports(self, file_id: int, imports: List[ImportDef]) -> None:
        self._conn.executemany(
            """INSERT INTO ci_imports (file_id, module, name, alias, is_from, line_no)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [(file_id, i.module, i.name, i.alias, 1 if i.is_from else 0, i.line_no) for i in imports]
        )

    def bulk_insert_classes(self, file_id: int, classes: List[ClassDef]) -> List[ClassDef]:
        result = []
        for c in classes:
            cursor = self._conn.execute(
                """INSERT INTO ci_classes
                   (file_id, name, bases_json, decorators_json, docstring, line_start, line_end, is_dataclass)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (file_id, c.name, json.dumps(c.bases), json.dumps(c.decorators),
                 c.docstring, c.line_start, c.line_end, 1 if c.is_dataclass else 0)
            )
            c.class_id = cursor.lastrowid
            c.file_id = file_id
            result.append(c)
        return result

    def bulk_insert_functions(self, file_id: int, functions: List[FunctionDef]) -> List[FunctionDef]:
        result = []
        for fn in functions:
            cursor = self._conn.execute(
                """INSERT INTO ci_functions
                   (file_id, class_id, name, params_json, return_type, decorators_json,
                    docstring, line_start, line_end, is_property, is_static, complexity)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (file_id, fn.class_id, fn.name, json.dumps(fn.params), fn.return_type,
                 json.dumps(fn.decorators), fn.docstring, fn.line_start, fn.line_end,
                 1 if fn.is_property else 0, 1 if fn.is_static else 0, fn.complexity)
            )
            fn.func_id = cursor.lastrowid
            fn.file_id = file_id
            result.append(fn)
        return result

    def bulk_insert_attributes(self, file_id: int, attrs: List[AttributeAccess]) -> None:
        self._conn.executemany(
            """INSERT INTO ci_attribute_access (file_id, func_id, target, attr_name, access_type, line_no)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [(file_id, a.func_id, a.target, a.attr_name, a.access_type, a.line_no) for a in attrs]
        )

    def bulk_insert_signals(self, file_id: int, signals: List[SignalDef]) -> None:
        self._conn.executemany(
            """INSERT INTO ci_signals (file_id, class_id, name, param_types_json, line_no)
               VALUES (?, ?, ?, ?, ?)""",
            [(file_id, s.class_id, s.name, json.dumps(s.param_types), s.line_no) for s in signals]
        )

    def bulk_insert_connections(self, file_id: int, connections: List[ConnectionDef]) -> None:
        self._conn.executemany(
            """INSERT INTO ci_connections (file_id, signal_expr, slot_expr, line_no)
               VALUES (?, ?, ?, ?)""",
            [(file_id, c.signal_expr, c.slot_expr, c.line_no) for c in connections]
        )

    def bulk_insert_calls(self, file_id: int, calls: List[CallDef]) -> None:
        self._conn.executemany(
            """INSERT INTO ci_calls (file_id, func_id, callee_expr, line_no)
               VALUES (?, ?, ?, ?)""",
            [(file_id, c.func_id, c.callee_expr, c.line_no) for c in calls]
        )

    def bulk_insert_ui_widgets(self, ui_file: str, widgets: List[UIWidget]) -> None:
        self._conn.executemany(
            """INSERT INTO ci_ui_widgets (ui_file, widget_name, widget_class, parent_name)
               VALUES (?, ?, ?, ?)""",
            [(ui_file, w.widget_name, w.widget_class, w.parent_name) for w in widgets]
        )

    def bulk_insert_diagnostics(self, diagnostics: List[Diagnostic]) -> None:
        now = self._now()
        self._conn.executemany(
            """INSERT INTO ci_diagnostics
               (file_id, rule_id, severity, message, line_no, context, is_resolved, first_seen, last_seen)
               VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)""",
            [(d.file_id, d.rule_id, d.severity, d.message, d.line_no, d.context, now, now)
             for d in diagnostics]
        )

    # === Query operations ===

    def find_functions(self, name: Optional[str] = None, class_name: Optional[str] = None,
                       file_pattern: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        sql = """
            SELECT f.*, c.name as class_name, fi.rel_path
            FROM ci_functions f
            LEFT JOIN ci_classes c ON f.class_id = c.class_id
            JOIN ci_files fi ON f.file_id = fi.file_id
            WHERE 1=1
        """
        params: list = []

        if name:
            sql += " AND f.name LIKE ?"
            params.append(f"%{name}%")
        if class_name:
            sql += " AND c.name LIKE ?"
            params.append(f"%{class_name}%")
        if file_pattern:
            sql += " AND fi.rel_path LIKE ?"
            params.append(f"%{file_pattern}%")

        sql += " ORDER BY fi.rel_path, f.line_start LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._func_row_to_dict(row) for row in rows]

    def find_classes(self, name: Optional[str] = None, base_class: Optional[str] = None,
                     limit: int = 50) -> List[Dict[str, Any]]:
        sql = """
            SELECT c.*, fi.rel_path
            FROM ci_classes c
            JOIN ci_files fi ON c.file_id = fi.file_id
            WHERE 1=1
        """
        params: list = []

        if name:
            sql += " AND c.name LIKE ?"
            params.append(f"%{name}%")
        if base_class:
            sql += " AND c.bases_json LIKE ?"
            params.append(f'%"{base_class}"%')

        sql += " ORDER BY fi.rel_path, c.line_start LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [{
            'class_id': row["class_id"],
            'name': row["name"],
            'bases': json.loads(row["bases_json"]),
            'decorators': json.loads(row["decorators_json"]),
            'docstring': row["docstring"],
            'line_start': row["line_start"],
            'line_end': row["line_end"],
            'is_dataclass': bool(row["is_dataclass"]),
            'file': row["rel_path"],
        } for row in rows]

    def get_callers(self, function_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        # Find calls where the callee expression contains the function name
        rows = self._conn.execute(
            """SELECT ca.*, fi.rel_path, fn.name as caller_func, cl.name as caller_class
               FROM ci_calls ca
               JOIN ci_files fi ON ca.file_id = fi.file_id
               LEFT JOIN ci_functions fn ON ca.func_id = fn.func_id
               LEFT JOIN ci_classes cl ON fn.class_id = cl.class_id
               WHERE ca.callee_expr LIKE ?
               ORDER BY fi.rel_path, ca.line_no
               LIMIT ?""",
            (f"%{function_name}%", limit)
        ).fetchall()
        return [{
            'file': row["rel_path"],
            'line_no': row["line_no"],
            'callee_expr': row["callee_expr"],
            'caller_func': row["caller_func"],
            'caller_class': row["caller_class"],
        } for row in rows]

    def get_callees(self, func_id: int) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT ca.callee_expr, ca.line_no, fi.rel_path
               FROM ci_calls ca
               JOIN ci_files fi ON ca.file_id = fi.file_id
               WHERE ca.func_id = ?
               ORDER BY ca.line_no""",
            (func_id,)
        ).fetchall()
        return [{
            'callee_expr': row["callee_expr"],
            'line_no': row["line_no"],
            'file': row["rel_path"],
        } for row in rows]

    def get_signals(self, name: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        sql = """
            SELECT s.*, fi.rel_path, c.name as class_name
            FROM ci_signals s
            JOIN ci_files fi ON s.file_id = fi.file_id
            LEFT JOIN ci_classes c ON s.class_id = c.class_id
            WHERE 1=1
        """
        params: list = []
        if name:
            sql += " AND s.name LIKE ?"
            params.append(f"%{name}%")
        sql += " ORDER BY fi.rel_path, s.line_no LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        results = []
        for row in rows:
            signal_name = row["name"]
            # Find connections for this signal
            conns = self._conn.execute(
                """SELECT co.signal_expr, co.slot_expr, co.line_no, fi2.rel_path as conn_file
                   FROM ci_connections co
                   JOIN ci_files fi2 ON co.file_id = fi2.file_id
                   WHERE co.signal_expr LIKE ?""",
                (f"%{signal_name}%",)
            ).fetchall()
            results.append({
                'signal_id': row["signal_id"],
                'name': signal_name,
                'class_name': row["class_name"],
                'param_types': json.loads(row["param_types_json"]),
                'line_no': row["line_no"],
                'file': row["rel_path"],
                'connections': [{
                    'signal_expr': c["signal_expr"],
                    'slot_expr': c["slot_expr"],
                    'line_no': c["line_no"],
                    'file': c["conn_file"],
                } for c in conns],
            })
        return results

    def get_state_fields(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT a.attr_name, a.access_type, a.line_no, fi.rel_path, fn.name as func_name
               FROM ci_attribute_access a
               JOIN ci_files fi ON a.file_id = fi.file_id
               LEFT JOIN ci_functions fn ON a.func_id = fn.func_id
               WHERE a.target = 'self.state'
               ORDER BY a.attr_name, fi.rel_path, a.line_no"""
        ).fetchall()

        # Group by field name
        fields: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            attr = row["attr_name"]
            if attr not in fields:
                fields[attr] = {'field': attr, 'accesses': []}
            fields[attr]['accesses'].append({
                'file': row["rel_path"],
                'func': row["func_name"],
                'access_type': row["access_type"],
                'line_no': row["line_no"],
            })

        return list(fields.values())

    def get_widget_refs(self, widget_name: str) -> List[Dict[str, Any]]:
        # Check the widget exists in .ui
        ui_row = self._conn.execute(
            "SELECT * FROM ci_ui_widgets WHERE widget_name = ?", (widget_name,)
        ).fetchone()

        # Find attribute accesses referencing this widget
        rows = self._conn.execute(
            """SELECT a.*, fi.rel_path, fn.name as func_name
               FROM ci_attribute_access a
               JOIN ci_files fi ON a.file_id = fi.file_id
               LEFT JOIN ci_functions fn ON a.func_id = fn.func_id
               WHERE a.attr_name = ?
               ORDER BY fi.rel_path, a.line_no""",
            (widget_name,)
        ).fetchall()

        return {
            'widget_name': widget_name,
            'ui_definition': {
                'widget_class': ui_row["widget_class"],
                'parent': ui_row["parent_name"],
                'ui_file': ui_row["ui_file"],
            } if ui_row else None,
            'references': [{
                'file': row["rel_path"],
                'func': row["func_name"],
                'target': row["target"],
                'access_type': row["access_type"],
                'line_no': row["line_no"],
            } for row in rows],
        }

    def get_diagnostics(self, severity: Optional[str] = None, rule_id: Optional[str] = None,
                        file_pattern: Optional[str] = None, limit: int = 100) -> List[Diagnostic]:
        sql = """
            SELECT d.*, fi.rel_path
            FROM ci_diagnostics d
            JOIN ci_files fi ON d.file_id = fi.file_id
            WHERE d.is_resolved = 0
        """
        params: list = []

        if severity:
            sql += " AND d.severity = ?"
            params.append(severity)
        if rule_id:
            sql += " AND d.rule_id = ?"
            params.append(rule_id)
        if file_pattern:
            sql += " AND fi.rel_path LIKE ?"
            params.append(f"%{file_pattern}%")

        sql += " ORDER BY CASE d.severity WHEN 'error' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END, fi.rel_path, d.line_no LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [Diagnostic(
            diag_id=row["diag_id"],
            file_id=row["file_id"],
            rule_id=row["rule_id"],
            severity=row["severity"],
            message=row["message"],
            line_no=row["line_no"],
            context=row["context"],
            is_resolved=bool(row["is_resolved"]),
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
        ) for row in rows]

    def search_fts(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            rows = self._conn.execute(
                """SELECT rel_path, class_names, function_names, docstrings, rank
                   FROM ci_fts
                   WHERE ci_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (query, limit)
            ).fetchall()
        except sqlite3.OperationalError:
            # Bad FTS query — fall back to LIKE search on functions
            return self.find_functions(name=query, limit=limit)

        return [{
            'rel_path': row["rel_path"],
            'class_names': row["class_names"],
            'function_names': row["function_names"],
            'docstrings': row["docstrings"],
            'score': -row["rank"],
        } for row in rows]

    def get_file_summary(self, rel_path: str) -> Optional[Dict[str, Any]]:
        fdef = self.get_file_by_path(rel_path)
        if not fdef:
            return None

        fid = fdef.file_id

        classes = self._conn.execute(
            "SELECT * FROM ci_classes WHERE file_id = ? ORDER BY line_start", (fid,)
        ).fetchall()

        functions = self._conn.execute(
            """SELECT f.*, c.name as class_name FROM ci_functions f
               LEFT JOIN ci_classes c ON f.class_id = c.class_id
               WHERE f.file_id = ? ORDER BY f.line_start""", (fid,)
        ).fetchall()

        imports = self._conn.execute(
            "SELECT * FROM ci_imports WHERE file_id = ? ORDER BY line_no", (fid,)
        ).fetchall()

        diagnostics = self._conn.execute(
            "SELECT * FROM ci_diagnostics WHERE file_id = ? AND is_resolved = 0 ORDER BY line_no", (fid,)
        ).fetchall()

        return {
            'file': fdef.to_dict(),
            'classes': [{
                'name': c["name"],
                'bases': json.loads(c["bases_json"]),
                'line_start': c["line_start"],
                'line_end': c["line_end"],
                'is_dataclass': bool(c["is_dataclass"]),
            } for c in classes],
            'functions': [{
                'name': f["name"],
                'class_name': f["class_name"],
                'params': json.loads(f["params_json"]),
                'return_type': f["return_type"],
                'line_start': f["line_start"],
                'line_end': f["line_end"],
                'is_property': bool(f["is_property"]),
                'complexity': f["complexity"],
            } for f in functions],
            'imports': [{
                'module': i["module"],
                'name': i["name"],
                'alias': i["alias"],
                'is_from': bool(i["is_from"]),
                'line_no': i["line_no"],
            } for i in imports],
            'diagnostics': [{
                'rule_id': d["rule_id"],
                'severity': d["severity"],
                'message': d["message"],
                'line_no': d["line_no"],
            } for d in diagnostics],
        }

    # === FTS management ===

    def update_fts_for_file(self, file_id: int, rel_path: str, class_names: str,
                            function_names: str, docstrings: str) -> None:
        """Update the FTS index for a file."""
        self._conn.execute("DELETE FROM ci_fts WHERE rel_path = ?", (rel_path,))
        self._conn.execute(
            "INSERT INTO ci_fts (rel_path, class_names, function_names, docstrings) VALUES (?, ?, ?, ?)",
            (rel_path, class_names, function_names, docstrings)
        )

    # === Knowledge cache ===

    def set_knowledge(self, key: str, value: Any) -> None:
        self._conn.execute(
            """INSERT INTO ci_knowledge (key, value_json, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at""",
            (key, json.dumps(value), self._now())
        )

    def get_knowledge(self, key: str) -> Optional[KnowledgeEntry]:
        row = self._conn.execute(
            "SELECT * FROM ci_knowledge WHERE key = ?", (key,)
        ).fetchone()
        if row:
            return KnowledgeEntry(
                key=row["key"],
                value=json.loads(row["value_json"]),
                updated_at=row["updated_at"],
            )
        return None

    def delete_knowledge(self, key: str) -> bool:
        cursor = self._conn.execute("DELETE FROM ci_knowledge WHERE key = ?", (key,))
        return cursor.rowcount > 0

    # === Stats & maintenance ===

    def get_stats(self) -> IndexStats:
        def _count(table: str) -> int:
            return self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        diag_counts = {}
        for row in self._conn.execute(
            "SELECT severity, COUNT(*) as cnt FROM ci_diagnostics WHERE is_resolved = 0 GROUP BY severity"
        ).fetchall():
            diag_counts[row["severity"]] = row["cnt"]

        parse_errors = self._conn.execute(
            "SELECT COUNT(*) FROM ci_files WHERE parse_error IS NOT NULL"
        ).fetchone()[0]

        return IndexStats(
            total_files=_count("ci_files"),
            total_classes=_count("ci_classes"),
            total_functions=_count("ci_functions"),
            total_imports=_count("ci_imports"),
            total_signals=_count("ci_signals"),
            total_connections=_count("ci_connections"),
            total_calls=_count("ci_calls"),
            total_ui_widgets=_count("ci_ui_widgets"),
            total_diagnostics=_count("ci_diagnostics"),
            errors=diag_counts.get("error", 0),
            warnings=diag_counts.get("warning", 0),
            info=diag_counts.get("info", 0),
            parse_errors=parse_errors,
        )

    def clear_diagnostics(self) -> None:
        self._conn.execute("DELETE FROM ci_diagnostics")

    def clear_ui_widgets(self) -> None:
        self._conn.execute("DELETE FROM ci_ui_widgets")

    def close(self) -> None:
        self._conn.close()

    # === Helpers ===

    def _func_row_to_dict(self, row) -> Dict[str, Any]:
        return {
            'func_id': row["func_id"],
            'name': row["name"],
            'class_name': row["class_name"],
            'params': json.loads(row["params_json"]),
            'return_type': row["return_type"],
            'decorators': json.loads(row["decorators_json"]),
            'docstring': row["docstring"],
            'line_start': row["line_start"],
            'line_end': row["line_end"],
            'is_property': bool(row["is_property"]),
            'is_static': bool(row["is_static"]),
            'complexity': row["complexity"],
            'file': row["rel_path"],
        }
