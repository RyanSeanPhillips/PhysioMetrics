"""
Code index service — AST parsing, file crawling, static analysis.

Standalone: no PyQt6 imports, usable from CLI/MCP/tests.
"""

import ast
import hashlib
import os
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Set, Tuple

from core.domain.code_index.models import (
    FileDef, ImportDef, ClassDef, FunctionDef, AttributeAccess,
    SignalDef, ConnectionDef, CallDef, UIWidget, Diagnostic,
    IndexStats,
)
from core.ports.code_index_port import CodeIndexPort


# ---------------------------------------------------------------------------
# AST Visitor
# ---------------------------------------------------------------------------

class CodeIndexVisitor(ast.NodeVisitor):
    """Extract structured information from a Python AST."""

    def __init__(self):
        self.imports: List[ImportDef] = []
        self.classes: List[ClassDef] = []
        self.functions: List[FunctionDef] = []
        self.attributes: List[AttributeAccess] = []
        self.signals: List[SignalDef] = []
        self.connections: List[ConnectionDef] = []
        self.calls: List[CallDef] = []

        # State tracking
        self._current_class: Optional[ClassDef] = None
        self._current_func: Optional[FunctionDef] = None
        self._class_stack: List[Optional[ClassDef]] = []
        self._func_stack: List[Optional[FunctionDef]] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(ImportDef(
                module=alias.name,
                alias=alias.asname,
                is_from=False,
                line_no=node.lineno,
            ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        if node.names:
            for alias in node.names:
                self.imports.append(ImportDef(
                    module=module,
                    name=alias.name,
                    alias=alias.asname,
                    is_from=True,
                    line_no=node.lineno,
                ))
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        bases = []
        for base in node.bases:
            bases.append(_unparse_node(base))

        decorators = [_unparse_node(d) for d in node.decorator_list]
        is_dataclass = any("dataclass" in d for d in decorators)

        docstring = ast.get_docstring(node)

        cls = ClassDef(
            name=node.name,
            bases=bases,
            decorators=decorators,
            docstring=docstring[:500] if docstring else None,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            is_dataclass=is_dataclass,
        )
        self.classes.append(cls)

        # Push class context
        self._class_stack.append(self._current_class)
        self._current_class = cls
        self.generic_visit(node)
        self._current_class = self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._process_function(node)

    def _process_function(self, node):
        params = []
        for arg in node.args.args:
            if arg.arg == 'self' or arg.arg == 'cls':
                continue
            param = {'name': arg.arg}
            if arg.annotation:
                param['type'] = _unparse_node(arg.annotation)
            params.append(param)

        # Defaults — align from the right
        defaults = node.args.defaults
        if defaults:
            offset = len(node.args.args) - len(defaults)
            # Adjust for self/cls
            self_offset = 1 if node.args.args and node.args.args[0].arg in ('self', 'cls') else 0
            for i, default in enumerate(defaults):
                param_idx = offset + i - self_offset
                if 0 <= param_idx < len(params):
                    params[param_idx]['default'] = _unparse_node(default)

        return_type = _unparse_node(node.returns) if node.returns else None
        decorators = [_unparse_node(d) for d in node.decorator_list]
        is_property = any("property" in d for d in decorators)
        is_static = any("staticmethod" in d for d in decorators)
        docstring = ast.get_docstring(node)

        complexity = _cyclomatic_complexity(node)

        func = FunctionDef(
            name=node.name,
            params=params,
            return_type=return_type,
            decorators=decorators,
            docstring=docstring[:500] if docstring else None,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            is_property=is_property,
            is_static=is_static,
            complexity=complexity,
        )

        # Link to current class
        if self._current_class is not None:
            func._pending_class = self._current_class

        self.functions.append(func)

        # Push function context
        self._func_stack.append(self._current_func)
        self._current_func = func
        self.generic_visit(node)
        self._current_func = self._func_stack.pop()

    def visit_Assign(self, node: ast.Assign):
        # Detect pyqtSignal declarations: name = pyqtSignal(...)
        if (len(node.targets) == 1
                and isinstance(node.value, ast.Call)
                and isinstance(node.targets[0], ast.Name)):

            call_name = _unparse_node(node.value.func)
            if call_name in ('pyqtSignal', 'Signal'):
                param_types = [_unparse_node(a) for a in node.value.args]
                sig = SignalDef(
                    name=node.targets[0].id,
                    param_types=param_types,
                    line_no=node.lineno,
                )
                if self._current_class:
                    sig._pending_class = self._current_class
                self.signals.append(sig)

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        callee = _unparse_node(node.func)

        # Detect .connect() calls
        if callee.endswith('.connect') and len(node.args) >= 1:
            signal_expr = callee[:-8]  # strip '.connect'
            slot_expr = _unparse_node(node.args[0])
            self.connections.append(ConnectionDef(
                signal_expr=signal_expr,
                slot_expr=slot_expr,
                line_no=node.lineno,
            ))
        else:
            call = CallDef(
                callee_expr=callee,
                line_no=node.lineno,
            )
            if self._current_func:
                call._pending_func = self._current_func
            self.calls.append(call)

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # Track self.X, self.state.X, self.mw.X patterns
        chain = _attribute_chain(node)
        if chain and chain[0] == 'self' and len(chain) >= 2:
            if len(chain) == 2:
                # self.X
                target = 'self'
                attr_name = chain[1]
            elif chain[1] in ('state', 'mw') and len(chain) >= 3:
                # self.state.X or self.mw.X
                target = f'self.{chain[1]}'
                attr_name = chain[2]
            else:
                # self.something_else.X — skip
                self.generic_visit(node)
                return

            # Determine access type
            access_type = _determine_access_type(node)

            attr = AttributeAccess(
                target=target,
                attr_name=attr_name,
                access_type=access_type,
                line_no=node.lineno,
            )
            if self._current_func:
                attr._pending_func = self._current_func
            self.attributes.append(attr)

        self.generic_visit(node)


# ---------------------------------------------------------------------------
# AST Helpers
# ---------------------------------------------------------------------------

def _unparse_node(node) -> str:
    """Convert an AST node to a string representation."""
    try:
        return ast.unparse(node)
    except Exception:
        return "<unknown>"


def _attribute_chain(node: ast.Attribute) -> Optional[List[str]]:
    """Extract the chain of attribute names: self.state.foo -> ['self', 'state', 'foo']."""
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        parts.reverse()
        return parts
    return None


def _determine_access_type(node: ast.Attribute) -> str:
    """Determine if an attribute access is read, write, call, or connect."""
    parent = getattr(node, '_parent', None)
    if parent is None:
        return "read"

    if isinstance(parent, ast.Call) and parent.func is node:
        return "call"

    if isinstance(parent, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        # Check if node is on the left side
        if isinstance(parent, ast.Assign):
            for target in parent.targets:
                if target is node:
                    return "write"
        elif isinstance(parent, ast.AugAssign) and parent.target is node:
            return "write"
        elif isinstance(parent, ast.AnnAssign) and parent.target is node:
            return "write"

    return "read"


def _cyclomatic_complexity(node: ast.FunctionDef) -> int:
    """Count cyclomatic complexity: 1 + branches."""
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            # Each and/or adds a branch
            complexity += len(child.values) - 1
    return complexity


def _set_parents(tree: ast.AST):
    """Annotate each AST node with a _parent reference."""
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node


# ---------------------------------------------------------------------------
# UI XML Parser
# ---------------------------------------------------------------------------

def parse_ui_file(ui_path: Path) -> List[UIWidget]:
    """Parse a Qt Designer .ui file and extract widget definitions."""
    widgets = []
    try:
        tree = ET.parse(str(ui_path))
        root = tree.getroot()
        ui_file = ui_path.name

        def _walk_widgets(element, parent_name=None):
            if element.tag == 'widget':
                widget_class = element.get('class', '')
                widget_name = element.get('name', '')
                if widget_name:
                    widgets.append(UIWidget(
                        ui_file=ui_file,
                        widget_name=widget_name,
                        widget_class=widget_class,
                        parent_name=parent_name,
                    ))
                    parent_name = widget_name

            # Also capture actions
            if element.tag == 'action':
                action_name = element.get('name', '')
                if action_name:
                    widgets.append(UIWidget(
                        ui_file=ui_file,
                        widget_name=action_name,
                        widget_class='QAction',
                        parent_name=parent_name,
                    ))

            for child in element:
                _walk_widgets(child, parent_name)

        _walk_widgets(root)
    except Exception:
        pass  # Non-fatal: missing/invalid .ui file

    return widgets


# ---------------------------------------------------------------------------
# File Crawler
# ---------------------------------------------------------------------------

SKIP_DIRS = {
    '__pycache__', '.git', '.venv', 'venv', 'node_modules',
    'build', 'dist', '.eggs', '*.egg-info', '_internal',
    'lib', 'dev_testing', 'old', '.mypy_cache', '.pytest_cache',
}


def compute_file_hash(path: Path) -> str:
    """SHA-256 hash of file contents."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def crawl_python_files(root: Path) -> List[Tuple[Path, str]]:
    """Walk the project and return (abs_path, rel_path) for each .py file."""
    results = []
    root = root.resolve()

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip excluded directories
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            if fname.endswith('.py'):
                abs_path = Path(dirpath) / fname
                rel_path = abs_path.relative_to(root).as_posix()
                results.append((abs_path, rel_path))

    return results


# ---------------------------------------------------------------------------
# Static Analyzer
# ---------------------------------------------------------------------------

class StaticAnalyzer:
    """Run static analysis rules against the indexed code."""

    def __init__(self, db: CodeIndexPort, project_root: Path):
        self.db = db
        self.project_root = project_root

    def run_all(self) -> List[Diagnostic]:
        """Run all analysis rules. Returns diagnostics found."""
        self.db.clear_diagnostics()
        diagnostics: List[Diagnostic] = []

        diagnostics.extend(self._check_undef_state_fields())
        diagnostics.extend(self._check_undef_ui_widgets())
        diagnostics.extend(self._check_large_methods())
        diagnostics.extend(self._check_mw_references())
        diagnostics.extend(self._check_dead_functions())
        diagnostics.extend(self._check_missing_return_type())
        diagnostics.extend(self._check_hasattr_guards())
        diagnostics.extend(self._check_circular_imports())
        diagnostics.extend(self._check_connect_target_missing())
        diagnostics.extend(self._check_signal_param_mismatch())
        diagnostics.extend(self._check_findchild_name_missing())

        if diagnostics:
            self.db.bulk_insert_diagnostics(diagnostics)

        return diagnostics

    def _check_undef_state_fields(self) -> List[Diagnostic]:
        """UNDEF_STATE_FIELD: self.state.X where X not defined in AppState."""
        # Parse AppState field names from core/state.py
        state_fields = self._get_appstate_fields()
        if not state_fields:
            return []

        diagnostics = []
        # Query all self.state accesses
        from core.adapters.code_index_sqlite import CodeIndexSQLite
        if not isinstance(self.db, CodeIndexSQLite):
            return []

        rows = self.db._conn.execute(
            """SELECT a.attr_name, a.line_no, a.access_type, fi.rel_path, fi.file_id
               FROM ci_attribute_access a
               JOIN ci_files fi ON a.file_id = fi.file_id
               WHERE a.target = 'self.state'"""
        ).fetchall()

        for row in rows:
            attr = row["attr_name"]
            if attr not in state_fields and not attr.startswith('_'):
                diagnostics.append(Diagnostic(
                    file_id=row["file_id"],
                    rule_id="UNDEF_STATE_FIELD",
                    severity="error",
                    message=f"self.state.{attr} — field '{attr}' not defined in AppState",
                    line_no=row["line_no"],
                    context=row["rel_path"],
                ))

        return diagnostics

    def _get_appstate_fields(self) -> Set[str]:
        """Parse field names from core/state.py AppState dataclass."""
        state_path = self.project_root / "core" / "state.py"
        if not state_path.exists():
            return set()

        try:
            source = state_path.read_text(encoding='utf-8')
            tree = ast.parse(source)
        except Exception:
            return set()

        fields = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'AppState':
                for item in node.body:
                    # Annotated assignments: field_name: Type = ...
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        fields.add(item.target.id)
                    # Regular assignments (rare in dataclass)
                    elif isinstance(item, ast.Assign):
                        for t in item.targets:
                            if isinstance(t, ast.Name):
                                fields.add(t.id)
                    # Methods — add as "fields" accessible via self.state.method()
                    elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        fields.add(item.name)
                break

        return fields

    def _check_undef_ui_widgets(self) -> List[Diagnostic]:
        """UNDEF_UI_WIDGET: self.mw.X or self.X (in main.py) where X not in .ui and not a Python attr."""
        from core.adapters.code_index_sqlite import CodeIndexSQLite
        if not isinstance(self.db, CodeIndexSQLite):
            return []

        # Get all known widget names
        ui_widgets = set()
        for row in self.db._conn.execute("SELECT widget_name FROM ci_ui_widgets").fetchall():
            ui_widgets.add(row["widget_name"])

        if not ui_widgets:
            return []

        # Get Python-defined attributes (class-level assignments + methods) in main.py
        python_attrs = set()
        main_file = self.db.get_file_by_path("main.py")
        if main_file:
            for row in self.db._conn.execute(
                "SELECT name FROM ci_functions WHERE file_id = ?", (main_file.file_id,)
            ).fetchall():
                python_attrs.add(row["name"])
            for row in self.db._conn.execute(
                "SELECT name FROM ci_classes WHERE file_id = ?", (main_file.file_id,)
            ).fetchall():
                python_attrs.add(row["name"])

        # Known non-widget attributes (common Qt/Python patterns)
        known_attrs = {
            'centralWidget', 'menuBar', 'statusBar', 'layout', 'parent',
            'setLayout', 'show', 'hide', 'close', 'update', 'repaint',
            'resize', 'move', 'setWindowTitle', 'setStyleSheet',
            'setCentralWidget', 'addWidget', 'removeWidget', 'sender',
        }
        all_known = ui_widgets | python_attrs | known_attrs

        diagnostics = []
        rows = self.db._conn.execute(
            """SELECT a.attr_name, a.line_no, fi.rel_path, fi.file_id
               FROM ci_attribute_access a
               JOIN ci_files fi ON a.file_id = fi.file_id
               WHERE a.target = 'self.mw'
               AND a.attr_name NOT IN (SELECT widget_name FROM ci_ui_widgets)"""
        ).fetchall()

        for row in rows:
            attr = row["attr_name"]
            if attr not in all_known and not attr.startswith('_'):
                diagnostics.append(Diagnostic(
                    file_id=row["file_id"],
                    rule_id="UNDEF_UI_WIDGET",
                    severity="warning",
                    message=f"self.mw.{attr} — '{attr}' not found in .ui file or Python definitions",
                    line_no=row["line_no"],
                    context=row["rel_path"],
                ))

        return diagnostics

    def _check_large_methods(self) -> List[Diagnostic]:
        """LARGE_METHOD: method > 50 lines or complexity > 15."""
        from core.adapters.code_index_sqlite import CodeIndexSQLite
        if not isinstance(self.db, CodeIndexSQLite):
            return []

        diagnostics = []
        rows = self.db._conn.execute(
            """SELECT f.func_id, f.name, f.line_start, f.line_end, f.complexity,
                      c.name as class_name, fi.rel_path, fi.file_id
               FROM ci_functions f
               LEFT JOIN ci_classes c ON f.class_id = c.class_id
               JOIN ci_files fi ON f.file_id = fi.file_id
               WHERE (f.line_end - f.line_start) > 50 OR f.complexity > 15"""
        ).fetchall()

        for row in rows:
            lines = row["line_end"] - row["line_start"]
            cx = row["complexity"]
            parts = []
            if lines > 50:
                parts.append(f"{lines} lines")
            if cx > 15:
                parts.append(f"complexity {cx}")

            qual_name = f"{row['class_name']}.{row['name']}" if row["class_name"] else row["name"]
            diagnostics.append(Diagnostic(
                file_id=row["file_id"],
                rule_id="LARGE_METHOD",
                severity="warning",
                message=f"{qual_name}: {', '.join(parts)}",
                line_no=row["line_start"],
                context=row["rel_path"],
            ))

        return diagnostics

    def _check_mw_references(self) -> List[Diagnostic]:
        """MW_REFERENCE: self.mw usage outside main.py (legacy anti-pattern)."""
        from core.adapters.code_index_sqlite import CodeIndexSQLite
        if not isinstance(self.db, CodeIndexSQLite):
            return []

        diagnostics = []
        rows = self.db._conn.execute(
            """SELECT DISTINCT fi.rel_path, fi.file_id, COUNT(*) as count
               FROM ci_attribute_access a
               JOIN ci_files fi ON a.file_id = fi.file_id
               WHERE a.target = 'self.mw'
               AND fi.rel_path != 'main.py'
               GROUP BY fi.file_id"""
        ).fetchall()

        for row in rows:
            diagnostics.append(Diagnostic(
                file_id=row["file_id"],
                rule_id="MW_REFERENCE",
                severity="info",
                message=f"self.mw used {row['count']} times (legacy pattern)",
                line_no=None,
                context=row["rel_path"],
            ))

        return diagnostics

    def _check_dead_functions(self) -> List[Diagnostic]:
        """DEAD_FUNCTION: function never referenced in any call."""
        from core.adapters.code_index_sqlite import CodeIndexSQLite
        if not isinstance(self.db, CodeIndexSQLite):
            return []

        # Build set of all referenced function names from calls and connections
        referenced: Set[str] = set()
        for row in self.db._conn.execute("SELECT DISTINCT callee_expr FROM ci_calls").fetchall():
            # Extract the last segment: 'self.foo' -> 'foo', 'module.bar' -> 'bar'
            expr = row["callee_expr"]
            parts = expr.rsplit('.', 1)
            referenced.add(parts[-1])
            referenced.add(expr)  # Also keep full expression

        for row in self.db._conn.execute("SELECT DISTINCT slot_expr FROM ci_connections").fetchall():
            expr = row["slot_expr"]
            parts = expr.rsplit('.', 1)
            referenced.add(parts[-1])
            referenced.add(expr)

        # Qt overrides that are called by the framework
        qt_overrides = {
            'paintEvent', 'resizeEvent', 'closeEvent', 'keyPressEvent',
            'mousePressEvent', 'mouseReleaseEvent', 'mouseMoveEvent',
            'wheelEvent', 'showEvent', 'hideEvent', 'eventFilter',
            'sizeHint', 'minimumSizeHint', 'contextMenuEvent',
            'dragEnterEvent', 'dropEvent', 'focusInEvent', 'focusOutEvent',
            'timerEvent', 'changeEvent', 'enterEvent', 'leaveEvent',
        }

        diagnostics = []
        rows = self.db._conn.execute(
            """SELECT f.func_id, f.name, f.line_start, c.name as class_name,
                      fi.rel_path, fi.file_id
               FROM ci_functions f
               LEFT JOIN ci_classes c ON f.class_id = c.class_id
               JOIN ci_files fi ON f.file_id = fi.file_id
               WHERE f.is_property = 0"""
        ).fetchall()

        for row in rows:
            name = row["name"]
            if name.startswith('_') or name in qt_overrides:
                continue
            if name in referenced:
                continue

            qual_name = f"{row['class_name']}.{name}" if row["class_name"] else name
            diagnostics.append(Diagnostic(
                file_id=row["file_id"],
                rule_id="DEAD_FUNCTION",
                severity="info",
                message=f"{qual_name} -- never called or connected",
                line_no=row["line_start"],
                context=row["rel_path"],
            ))

        return diagnostics

    def _check_missing_return_type(self) -> List[Diagnostic]:
        """MISSING_RETURN_TYPE: public method with no return type annotation."""
        from core.adapters.code_index_sqlite import CodeIndexSQLite
        if not isinstance(self.db, CodeIndexSQLite):
            return []

        diagnostics = []
        rows = self.db._conn.execute(
            """SELECT f.name, f.line_start, c.name as class_name, fi.rel_path, fi.file_id
               FROM ci_functions f
               LEFT JOIN ci_classes c ON f.class_id = c.class_id
               JOIN ci_files fi ON f.file_id = fi.file_id
               WHERE f.return_type IS NULL
               AND f.name NOT LIKE '\\_%' ESCAPE '\\'
               AND f.is_property = 0
               AND f.class_id IS NOT NULL"""
        ).fetchall()

        for row in rows:
            name = row["name"]
            if name.startswith('__'):
                continue
            qual_name = f"{row['class_name']}.{name}" if row["class_name"] else name
            diagnostics.append(Diagnostic(
                file_id=row["file_id"],
                rule_id="MISSING_RETURN_TYPE",
                severity="info",
                message=f"{qual_name} — no return type annotation",
                line_no=row["line_start"],
                context=row["rel_path"],
            ))

        return diagnostics

    def _check_hasattr_guards(self) -> List[Diagnostic]:
        """HASATTR_GUARD: hasattr(self, 'X') indicating initialization uncertainty."""
        from core.adapters.code_index_sqlite import CodeIndexSQLite
        if not isinstance(self.db, CodeIndexSQLite):
            return []

        diagnostics = []
        rows = self.db._conn.execute(
            """SELECT ca.callee_expr, ca.line_no, fi.rel_path, fi.file_id
               FROM ci_calls ca
               JOIN ci_files fi ON ca.file_id = fi.file_id
               WHERE ca.callee_expr = 'hasattr'"""
        ).fetchall()

        for row in rows:
            diagnostics.append(Diagnostic(
                file_id=row["file_id"],
                rule_id="HASATTR_GUARD",
                severity="info",
                message="hasattr() guard — may indicate initialization uncertainty",
                line_no=row["line_no"],
                context=row["rel_path"],
            ))

        return diagnostics

    def _check_circular_imports(self) -> List[Diagnostic]:
        """CIRCULAR_IMPORT: Module A imports B and B imports A."""
        from core.adapters.code_index_sqlite import CodeIndexSQLite
        if not isinstance(self.db, CodeIndexSQLite):
            return []

        diagnostics = []

        # Build import graph: file -> set of imported modules
        rows = self.db._conn.execute(
            """SELECT fi.rel_path, fi.file_id, i.module
               FROM ci_imports i
               JOIN ci_files fi ON i.file_id = fi.file_id"""
        ).fetchall()

        # Map rel_path -> set of modules imported
        file_imports: Dict[str, Set[str]] = {}
        file_ids: Dict[str, int] = {}
        for row in rows:
            rp = row["rel_path"]
            if rp not in file_imports:
                file_imports[rp] = set()
                file_ids[rp] = row["file_id"]
            file_imports[rp].add(row["module"])

        # Map module name -> rel_path
        module_to_path: Dict[str, str] = {}
        for rp in file_imports:
            # Convert rel_path to module: 'core/state.py' -> 'core.state'
            mod = rp.replace('/', '.').replace('\\', '.')
            if mod.endswith('.py'):
                mod = mod[:-3]
            module_to_path[mod] = rp

        # Check for circular pairs
        seen_pairs: Set[Tuple[str, str]] = set()
        for rp_a, imports_a in file_imports.items():
            mod_a = rp_a.replace('/', '.').replace('\\', '.').removesuffix('.py')
            for imp_mod in imports_a:
                rp_b = module_to_path.get(imp_mod)
                if rp_b and rp_b in file_imports:
                    # Does B import A?
                    imports_b = file_imports[rp_b]
                    if mod_a in imports_b:
                        pair = tuple(sorted([rp_a, rp_b]))
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            diagnostics.append(Diagnostic(
                                file_id=file_ids[rp_a],
                                rule_id="CIRCULAR_IMPORT",
                                severity="warning",
                                message=f"Circular import: {rp_a} <-> {rp_b}",
                                context=rp_a,
                            ))

        return diagnostics

    def _check_connect_target_missing(self) -> List[Diagnostic]:
        """CONNECT_TARGET_MISSING: .connect(self.method) where method doesn't exist."""
        from core.adapters.code_index_sqlite import CodeIndexSQLite
        if not isinstance(self.db, CodeIndexSQLite):
            return []

        # Build set of all known function names (any class, any file)
        all_func_names: Set[str] = set()
        for row in self.db._conn.execute(
            "SELECT DISTINCT name FROM ci_functions"
        ).fetchall():
            all_func_names.add(row["name"])

        # Also include signal names as valid connect targets (for .emit)
        for row in self.db._conn.execute(
            "SELECT DISTINCT name FROM ci_signals"
        ).fetchall():
            all_func_names.add(row["name"])

        # Build per-file function names for precise checking
        file_funcs: Dict[int, Set[str]] = {}
        for row in self.db._conn.execute(
            "SELECT file_id, name FROM ci_functions"
        ).fetchall():
            file_funcs.setdefault(row["file_id"], set()).add(row["name"])

        # main.py functions for self.mw.X resolution
        main_file = self.db.get_file_by_path("main.py")
        main_funcs = file_funcs.get(main_file.file_id, set()) if main_file else set()

        # Common Qt/Python builtins that won't be in our index
        qt_builtins = {
            # QWidget
            'close', 'show', 'hide', 'update', 'repaint', 'raise_', 'lower',
            'setEnabled', 'setDisabled', 'setVisible', 'setHidden',
            'setFocus', 'clearFocus', 'setStyleSheet', 'setToolTip',
            'setWindowTitle', 'resize', 'move', 'adjustSize', 'activateWindow',
            # QDialog
            'accept', 'reject', 'done', 'exec', 'open',
            # QAbstractButton / QLineEdit / QTextEdit
            'click', 'toggle', 'setChecked', 'animateClick',
            'clear', 'setText', 'setPlainText', 'setHtml',
            'selectAll', 'copy', 'cut', 'paste', 'undo', 'redo',
            # QComboBox
            'setCurrentIndex', 'setCurrentText',
            # QAbstractItemView
            'reset', 'scrollToTop', 'scrollToBottom',
            # QObject / QTimer / QThread
            'deleteLater', 'setObjectName', 'start', 'stop',
            'quit', 'terminate', 'wait',
            # QApplication
            'exit', 'processEvents',
            # Signal
            'emit',
            # QAction
            'trigger',
            # QProgressBar / QSlider
            'setValue', 'setRange', 'setMaximum', 'setMinimum',
            # QTabWidget / QStackedWidget
            'setCurrentWidget',
            # Layout
            'addWidget', 'removeWidget', 'addItem',
            # Common Python
            'append', 'insert', 'remove', 'pop', 'sort', 'extend',
            'setData', 'setModel', 'model',
            # Scroll
            'ensureVisible', 'ensureWidgetVisible',
        }

        # Also allow all widget names from .ui as valid targets
        ui_widgets = set()
        for row in self.db._conn.execute("SELECT widget_name FROM ci_ui_widgets").fetchall():
            ui_widgets.add(row["widget_name"])

        diagnostics = []
        rows = self.db._conn.execute(
            """SELECT co.slot_expr, co.signal_expr, co.line_no, fi.rel_path, fi.file_id
               FROM ci_connections co
               JOIN ci_files fi ON co.file_id = fi.file_id"""
        ).fetchall()

        for row in rows:
            slot = row["slot_expr"]

            # Skip lambdas, partials, and complex expressions
            if any(kw in slot for kw in ('lambda', 'partial', '(', '[', '{')):
                continue

            # Extract method name from slot expression
            parts = slot.rsplit('.', 1)
            if len(parts) != 2:
                continue

            prefix, method_name = parts

            # Skip Qt builtins and private methods starting with __
            if method_name in qt_builtins or method_name.startswith('__'):
                continue

            if prefix == 'self':
                # Check same file, then globally
                file_fns = file_funcs.get(row["file_id"], set())
                if method_name not in file_fns and method_name not in all_func_names:
                    diagnostics.append(Diagnostic(
                        file_id=row["file_id"],
                        rule_id="CONNECT_TARGET_MISSING",
                        severity="error",
                        message=f".connect({slot}) — method '{method_name}' not found",
                        line_no=row["line_no"],
                        context=row["rel_path"],
                    ))
            elif prefix == 'self.mw':
                # Check main.py functions
                if (method_name not in main_funcs
                        and method_name not in all_func_names
                        and method_name not in ui_widgets):
                    diagnostics.append(Diagnostic(
                        file_id=row["file_id"],
                        rule_id="CONNECT_TARGET_MISSING",
                        severity="warning",
                        message=f".connect({slot}) — '{method_name}' not found on MainWindow",
                        line_no=row["line_no"],
                        context=row["rel_path"],
                    ))

        return diagnostics

    def _check_signal_param_mismatch(self) -> List[Diagnostic]:
        """SIGNAL_PARAM_MISMATCH: slot requires more params than signal provides."""
        from core.adapters.code_index_sqlite import CodeIndexSQLite
        if not isinstance(self.db, CodeIndexSQLite):
            return []

        # Build signal name -> param count map
        # If a signal name is declared with different param counts, track all
        signal_param_counts: Dict[str, Set[int]] = {}
        for row in self.db._conn.execute(
            "SELECT name, param_types_json FROM ci_signals"
        ).fetchall():
            types = json.loads(row["param_types_json"]) if row["param_types_json"] else []
            signal_param_counts.setdefault(row["name"], set()).add(len(types))

        # Common Qt signals with known param counts
        qt_signal_params = {
            'clicked': {0, 1},        # clicked() or clicked(bool)
            'toggled': {1},           # toggled(bool)
            'triggered': {0, 1},     # triggered() or triggered(bool)
            'pressed': {0, 1},       # QAbstractButton: 0, QAbstractItemView: 1 (QModelIndex)
            'released': {0},
            'textChanged': {1},      # textChanged(str)
            'textEdited': {1},
            'valueChanged': {1},     # valueChanged(int) or valueChanged(float)
            'currentIndexChanged': {1},
            'currentTextChanged': {1},
            'stateChanged': {1},
            'activated': {1},
            'highlighted': {1},
            'finished': {0, 1},      # QDialog: finished(int), QThread: finished()
            'accepted': {0},
            'rejected': {0},
            'timeout': {0},
            'destroyed': {0, 1},
            'objectNameChanged': {1},
            'returnPressed': {0},
            'editingFinished': {0},
            'selectionChanged': {0},
            'cellClicked': {2},
            'cellDoubleClicked': {2},
            'itemClicked': {1},
            'itemDoubleClicked': {1},
            'itemSelectionChanged': {0},
            'currentChanged': {1, 2},
            'tabCloseRequested': {1},
            'customContextMenuRequested': {1},
            'sectionClicked': {1},
            'sectionResized': {3},
        }

        # Merge Qt builtins into our map
        for sig_name, counts in qt_signal_params.items():
            signal_param_counts.setdefault(sig_name, set()).update(counts)

        # Build function name -> set of required param counts
        # Group by (file_id, class_id) for precision, but also track globally
        func_params: Dict[str, List[Tuple[int, int]]] = {}  # name -> [(required, total)]
        for row in self.db._conn.execute(
            "SELECT name, params_json FROM ci_functions WHERE class_id IS NOT NULL"
        ).fetchall():
            params = json.loads(row["params_json"]) if row["params_json"] else []
            required = sum(1 for p in params if 'default' not in p)
            total = len(params)
            func_params.setdefault(row["name"], []).append((required, total))

        diagnostics = []
        rows = self.db._conn.execute(
            """SELECT co.signal_expr, co.slot_expr, co.line_no, fi.rel_path, fi.file_id
               FROM ci_connections co
               JOIN ci_files fi ON co.file_id = fi.file_id"""
        ).fetchall()

        for row in rows:
            signal_expr = row["signal_expr"]
            slot_expr = row["slot_expr"]

            # Skip lambdas/partials — they adapt param counts
            if any(kw in slot_expr for kw in ('lambda', 'partial', '(')):
                continue

            # Extract signal name (last component)
            sig_name = signal_expr.rsplit('.', 1)[-1]
            if sig_name not in signal_param_counts:
                continue

            # Max params the signal can provide
            sig_max = max(signal_param_counts[sig_name])

            # Extract slot method name
            slot_parts = slot_expr.rsplit('.', 1)
            if len(slot_parts) != 2:
                continue
            slot_name = slot_parts[-1]

            if slot_name not in func_params:
                continue

            # Check all overloads — flag if ALL overloads require more params than signal provides
            all_overloads_mismatch = True
            min_required = None
            for required, total in func_params[slot_name]:
                if min_required is None or required < min_required:
                    min_required = required
                if required <= sig_max:
                    all_overloads_mismatch = False
                    break

            if all_overloads_mismatch and min_required is not None and min_required > sig_max:
                diagnostics.append(Diagnostic(
                    file_id=row["file_id"],
                    rule_id="SIGNAL_PARAM_MISMATCH",
                    severity="warning",
                    message=(
                        f"{signal_expr}.connect({slot_expr}) — signal provides "
                        f"{sig_max} arg(s) but slot requires {min_required}"
                    ),
                    line_no=row["line_no"],
                    context=row["rel_path"],
                ))

        return diagnostics

    def _check_findchild_name_missing(self) -> List[Diagnostic]:
        """FINDCHILD_NAME_MISSING: findChild(QType, 'name') where name not in .ui widgets."""
        from core.adapters.code_index_sqlite import CodeIndexSQLite
        if not isinstance(self.db, CodeIndexSQLite):
            return []

        # Get all widget names from .ui
        ui_widgets = set()
        for row in self.db._conn.execute("SELECT widget_name FROM ci_ui_widgets").fetchall():
            ui_widgets.add(row["widget_name"])

        if not ui_widgets:
            return []

        # Find files that call findChild
        rows = self.db._conn.execute(
            """SELECT DISTINCT fi.file_id, fi.rel_path, fi.abs_path
               FROM ci_calls ca
               JOIN ci_files fi ON ca.file_id = fi.file_id
               WHERE ca.callee_expr LIKE '%findChild%'"""
        ).fetchall()

        diagnostics = []
        for row in rows:
            try:
                source = Path(row["abs_path"]).read_text(encoding='utf-8', errors='replace')
                tree = ast.parse(source)
            except Exception:
                continue

            # Walk AST for findChild(QType, "name") or findChild("name")
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == 'findChild'):
                    continue

                # Extract string argument (widget name)
                name_arg = None
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        name_arg = arg.value
                        break

                if name_arg and name_arg not in ui_widgets:
                    diagnostics.append(Diagnostic(
                        file_id=row["file_id"],
                        rule_id="FINDCHILD_NAME_MISSING",
                        severity="warning",
                        message=f"findChild(..., '{name_arg}') — '{name_arg}' not found in .ui file",
                        line_no=node.lineno,
                        context=row["rel_path"],
                    ))

        return diagnostics


# ---------------------------------------------------------------------------
# Code Index Service
# ---------------------------------------------------------------------------

class CodeIndexService:
    """Main service for building and querying the code index."""

    def __init__(self, db: CodeIndexPort, project_root: Path):
        self.db = db
        self.project_root = project_root.resolve()
        self.analyzer = StaticAnalyzer(db, self.project_root)

    def full_rebuild(self, run_analysis: bool = True) -> IndexStats:
        """Full rebuild: crawl all files, parse ASTs, index everything."""
        t0 = time.time()

        # Clear all data for a clean rebuild
        from core.adapters.code_index_sqlite import CodeIndexSQLite
        if isinstance(self.db, CodeIndexSQLite):
            self.db._conn.executescript("""
                DELETE FROM ci_fts;
                DELETE FROM ci_diagnostics;
                DELETE FROM ci_files;
                DELETE FROM ci_ui_widgets;
            """)

        # Parse UI file first
        ui_files = list(self.project_root.glob("ui/*.ui"))
        for ui_path in ui_files:
            widgets = parse_ui_file(ui_path)
            if widgets:
                self.db.bulk_insert_ui_widgets(ui_path.name, widgets)

        # Crawl and index Python files
        py_files = crawl_python_files(self.project_root)
        if isinstance(self.db, CodeIndexSQLite):
            with self.db._transaction():
                for abs_path, rel_path in py_files:
                    self._index_file(abs_path, rel_path)

        # Run static analysis
        if run_analysis:
            self.analyzer.run_all()

        elapsed = time.time() - t0
        stats = self.db.get_stats()

        # Cache rebuild info
        self.db.set_knowledge("last_rebuild", {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "files_indexed": stats.total_files,
        })

        return stats

    def incremental_update(self) -> Dict[str, int]:
        """Check file hashes and re-index only changed files."""
        py_files = crawl_python_files(self.project_root)
        changed = 0
        added = 0
        removed = 0

        # Build set of current files
        current_paths = {rel for _, rel in py_files}

        # Check for removed files
        for existing in self.db.list_files():
            if existing.rel_path not in current_paths:
                self.db.delete_file(existing.file_id)
                removed += 1

        # Check for new/changed files
        from core.adapters.code_index_sqlite import CodeIndexSQLite
        if isinstance(self.db, CodeIndexSQLite):
            with self.db._transaction():
                for abs_path, rel_path in py_files:
                    existing = self.db.get_file_by_path(rel_path)
                    file_hash = compute_file_hash(abs_path)

                    if existing is None:
                        self._index_file(abs_path, rel_path)
                        added += 1
                    elif existing.file_hash != file_hash:
                        # Re-index: CASCADE delete old data, then re-parse
                        self.db.delete_file(existing.file_id)
                        self._index_file(abs_path, rel_path)
                        changed += 1

        # Re-run analysis if anything changed
        if changed + added + removed > 0:
            self.analyzer.run_all()

        return {'changed': changed, 'added': added, 'removed': removed}

    def reindex_file(self, rel_path: str) -> bool:
        """Re-index a single file (called after edit). Returns True if successful."""
        abs_path = self.project_root / rel_path
        if not abs_path.exists():
            # File was deleted
            existing = self.db.get_file_by_path(rel_path)
            if existing:
                self.db.delete_file(existing.file_id)
            return True

        # Delete old data
        existing = self.db.get_file_by_path(rel_path)
        if existing:
            self.db.delete_file(existing.file_id)

        from core.adapters.code_index_sqlite import CodeIndexSQLite
        if isinstance(self.db, CodeIndexSQLite):
            with self.db._transaction():
                self._index_file(abs_path, rel_path)

        return True

    def _index_file(self, abs_path: Path, rel_path: str):
        """Parse and index a single Python file."""
        try:
            source = abs_path.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            self._insert_file_with_error(abs_path, rel_path, str(e))
            return

        file_hash = compute_file_hash(abs_path)
        line_count = source.count('\n') + 1
        mtime = datetime.fromtimestamp(abs_path.stat().st_mtime).isoformat()

        # Try to parse
        parse_error = None
        try:
            tree = ast.parse(source, filename=rel_path)
        except SyntaxError as e:
            parse_error = f"Line {e.lineno}: {e.msg}"
            tree = None

        # Insert file record
        fdef = self.db.upsert_file(FileDef(
            rel_path=rel_path,
            abs_path=str(abs_path),
            file_hash=file_hash,
            mtime=mtime,
            line_count=line_count,
            parse_error=parse_error,
            indexed_at=datetime.now().isoformat(),
        ))

        if tree is None:
            return

        # Set parent references for access type detection
        _set_parents(tree)

        # Visit AST
        visitor = CodeIndexVisitor()
        visitor.visit(tree)

        file_id = fdef.file_id

        # Insert imports
        if visitor.imports:
            self.db.bulk_insert_imports(file_id, visitor.imports)

        # Insert classes (need class_ids for functions)
        class_map: Dict[int, int] = {}  # id(ClassDef) -> class_id
        if visitor.classes:
            inserted_classes = self.db.bulk_insert_classes(file_id, visitor.classes)
            for cls in inserted_classes:
                class_map[id(cls)] = cls.class_id

        # Insert functions (link to classes, need func_ids for calls/attrs)
        func_map: Dict[int, int] = {}  # id(FunctionDef) -> func_id
        for fn in visitor.functions:
            pending_cls = getattr(fn, '_pending_class', None)
            if pending_cls is not None:
                fn.class_id = class_map.get(id(pending_cls))
            delattr(fn, '_pending_class') if hasattr(fn, '_pending_class') else None

        if visitor.functions:
            inserted_funcs = self.db.bulk_insert_functions(file_id, visitor.functions)
            for fn in inserted_funcs:
                func_map[id(fn)] = fn.func_id

        # Link signals to classes
        for sig in visitor.signals:
            pending_cls = getattr(sig, '_pending_class', None)
            if pending_cls is not None:
                sig.class_id = class_map.get(id(pending_cls))
            if hasattr(sig, '_pending_class'):
                delattr(sig, '_pending_class')

        # Link calls and attributes to functions
        for call in visitor.calls:
            pending_fn = getattr(call, '_pending_func', None)
            if pending_fn is not None:
                call.func_id = func_map.get(id(pending_fn))
            if hasattr(call, '_pending_func'):
                delattr(call, '_pending_func')

        for attr in visitor.attributes:
            pending_fn = getattr(attr, '_pending_func', None)
            if pending_fn is not None:
                attr.func_id = func_map.get(id(pending_fn))
            if hasattr(attr, '_pending_func'):
                delattr(attr, '_pending_func')

        # Bulk insert remaining
        if visitor.signals:
            self.db.bulk_insert_signals(file_id, visitor.signals)
        if visitor.connections:
            self.db.bulk_insert_connections(file_id, visitor.connections)
        if visitor.calls:
            self.db.bulk_insert_calls(file_id, visitor.calls)
        if visitor.attributes:
            self.db.bulk_insert_attributes(file_id, visitor.attributes)

        # Update FTS index
        class_names = " ".join(c.name for c in visitor.classes)
        func_names = " ".join(f.name for f in visitor.functions)
        docstrings = " ".join(
            (c.docstring or "") for c in visitor.classes
        ) + " " + " ".join(
            (f.docstring or "") for f in visitor.functions
        )
        self.db.update_fts_for_file(file_id, rel_path, class_names, func_names, docstrings.strip())

    def _insert_file_with_error(self, abs_path: Path, rel_path: str, error: str):
        """Insert a file record with a read error."""
        try:
            mtime = datetime.fromtimestamp(abs_path.stat().st_mtime).isoformat()
        except Exception:
            mtime = datetime.now().isoformat()

        self.db.upsert_file(FileDef(
            rel_path=rel_path,
            abs_path=str(abs_path),
            file_hash="",
            mtime=mtime,
            line_count=0,
            parse_error=error,
            indexed_at=datetime.now().isoformat(),
        ))

    # === Query pass-throughs ===

    def find_function(self, name: Optional[str] = None, class_name: Optional[str] = None,
                      file_pattern: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        return self.db.find_functions(name=name, class_name=class_name,
                                      file_pattern=file_pattern, limit=limit)

    def find_class(self, name: Optional[str] = None, base_class: Optional[str] = None,
                   limit: int = 50) -> List[Dict[str, Any]]:
        return self.db.find_classes(name=name, base_class=base_class, limit=limit)

    def get_callers(self, function_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        return self.db.get_callers(function_name, limit=limit)

    def get_callees(self, func_id: int) -> List[Dict[str, Any]]:
        return self.db.get_callees(func_id)

    def get_diagnostics(self, severity: Optional[str] = None, rule_id: Optional[str] = None,
                        file_pattern: Optional[str] = None, limit: int = 100) -> List[Diagnostic]:
        return self.db.get_diagnostics(severity=severity, rule_id=rule_id,
                                        file_pattern=file_pattern, limit=limit)

    def get_signals(self, name: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        return self.db.get_signals(name=name, limit=limit)

    def get_state_fields(self) -> List[Dict[str, Any]]:
        return self.db.get_state_fields()

    def get_widget_refs(self, widget_name: str) -> Any:
        return self.db.get_widget_refs(widget_name)

    def search_code(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        return self.db.search_fts(query, limit=limit)

    def get_file_summary(self, rel_path: str) -> Optional[Dict[str, Any]]:
        return self.db.get_file_summary(rel_path)

    def get_stats(self) -> IndexStats:
        return self.db.get_stats()

    def cache_knowledge(self, key: str, value: Any) -> None:
        self.db.set_knowledge(key, value)

    def get_knowledge(self, key: str) -> Any:
        entry = self.db.get_knowledge(key)
        return entry.value if entry else None
