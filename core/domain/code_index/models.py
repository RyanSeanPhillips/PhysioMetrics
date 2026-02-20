"""
Code index domain models.

Pure dataclasses for the code index system — no PyQt6 dependencies.
Each model maps to a SQLite table and supports dict serialization.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class FileDef:
    """A Python source file in the index."""
    file_id: int = 0
    rel_path: str = ""
    abs_path: str = ""
    file_hash: str = ""
    mtime: str = ""
    line_count: int = 0
    parse_error: Optional[str] = None
    indexed_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'file_id': self.file_id,
            'rel_path': self.rel_path,
            'abs_path': self.abs_path,
            'file_hash': self.file_hash,
            'mtime': self.mtime,
            'line_count': self.line_count,
            'parse_error': self.parse_error,
            'indexed_at': self.indexed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileDef':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ImportDef:
    """An import statement."""
    import_id: int = 0
    file_id: int = 0
    module: str = ""
    name: Optional[str] = None
    alias: Optional[str] = None
    is_from: bool = False
    line_no: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'import_id': self.import_id,
            'file_id': self.file_id,
            'module': self.module,
            'name': self.name,
            'alias': self.alias,
            'is_from': self.is_from,
            'line_no': self.line_no,
        }


@dataclass
class ClassDef:
    """A class definition."""
    class_id: int = 0
    file_id: int = 0
    name: str = ""
    bases: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    line_start: int = 0
    line_end: int = 0
    is_dataclass: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'class_id': self.class_id,
            'file_id': self.file_id,
            'name': self.name,
            'bases': self.bases,
            'decorators': self.decorators,
            'docstring': self.docstring,
            'line_start': self.line_start,
            'line_end': self.line_end,
            'is_dataclass': self.is_dataclass,
        }


@dataclass
class FunctionDef:
    """A function or method definition."""
    func_id: int = 0
    file_id: int = 0
    class_id: Optional[int] = None
    name: str = ""
    params: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    line_start: int = 0
    line_end: int = 0
    is_property: bool = False
    is_static: bool = False
    complexity: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'func_id': self.func_id,
            'file_id': self.file_id,
            'class_id': self.class_id,
            'name': self.name,
            'params': self.params,
            'return_type': self.return_type,
            'decorators': self.decorators,
            'docstring': self.docstring,
            'line_start': self.line_start,
            'line_end': self.line_end,
            'is_property': self.is_property,
            'is_static': self.is_static,
            'complexity': self.complexity,
        }


@dataclass
class AttributeAccess:
    """An attribute access on self.state, self.mw, etc."""
    access_id: int = 0
    file_id: int = 0
    func_id: Optional[int] = None
    target: str = ""       # 'self.state', 'self.mw', 'self'
    attr_name: str = ""
    access_type: str = "read"  # read/write/call/connect
    line_no: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'access_id': self.access_id,
            'file_id': self.file_id,
            'func_id': self.func_id,
            'target': self.target,
            'attr_name': self.attr_name,
            'access_type': self.access_type,
            'line_no': self.line_no,
        }


@dataclass
class SignalDef:
    """A pyqtSignal declaration."""
    signal_id: int = 0
    file_id: int = 0
    class_id: Optional[int] = None
    name: str = ""
    param_types: List[str] = field(default_factory=list)
    line_no: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'signal_id': self.signal_id,
            'file_id': self.file_id,
            'class_id': self.class_id,
            'name': self.name,
            'param_types': self.param_types,
            'line_no': self.line_no,
        }


@dataclass
class ConnectionDef:
    """A signal.connect() call."""
    conn_id: int = 0
    file_id: int = 0
    signal_expr: str = ""
    slot_expr: str = ""
    line_no: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'conn_id': self.conn_id,
            'file_id': self.file_id,
            'signal_expr': self.signal_expr,
            'slot_expr': self.slot_expr,
            'line_no': self.line_no,
        }


@dataclass
class CallDef:
    """A function call site."""
    call_id: int = 0
    file_id: int = 0
    func_id: Optional[int] = None
    callee_expr: str = ""
    line_no: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'call_id': self.call_id,
            'file_id': self.file_id,
            'func_id': self.func_id,
            'callee_expr': self.callee_expr,
            'line_no': self.line_no,
        }


@dataclass
class UIWidget:
    """A widget from a .ui file."""
    widget_id: int = 0
    ui_file: str = ""
    widget_name: str = ""
    widget_class: str = ""
    parent_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'widget_id': self.widget_id,
            'ui_file': self.ui_file,
            'widget_name': self.widget_name,
            'widget_class': self.widget_class,
            'parent_name': self.parent_name,
        }


@dataclass
class Diagnostic:
    """A static analysis diagnostic."""
    diag_id: int = 0
    file_id: int = 0
    rule_id: str = ""
    severity: str = "warning"
    message: str = ""
    line_no: Optional[int] = None
    context: Optional[str] = None
    is_resolved: bool = False
    first_seen: str = ""
    last_seen: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'diag_id': self.diag_id,
            'file_id': self.file_id,
            'rule_id': self.rule_id,
            'severity': self.severity,
            'message': self.message,
            'line_no': self.line_no,
            'context': self.context,
            'is_resolved': self.is_resolved,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
        }


@dataclass
class KnowledgeEntry:
    """A cached knowledge entry persisted across sessions."""
    key: str = ""
    value: Any = None
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'key': self.key,
            'value': self.value,
            'updated_at': self.updated_at,
        }


@dataclass
class IndexStats:
    """Summary statistics for the index."""
    total_files: int = 0
    total_classes: int = 0
    total_functions: int = 0
    total_imports: int = 0
    total_signals: int = 0
    total_connections: int = 0
    total_calls: int = 0
    total_ui_widgets: int = 0
    total_diagnostics: int = 0
    errors: int = 0
    warnings: int = 0
    info: int = 0
    parse_errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_files': self.total_files,
            'total_classes': self.total_classes,
            'total_functions': self.total_functions,
            'total_imports': self.total_imports,
            'total_signals': self.total_signals,
            'total_connections': self.total_connections,
            'total_calls': self.total_calls,
            'total_ui_widgets': self.total_ui_widgets,
            'total_diagnostics': self.total_diagnostics,
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info,
            'parse_errors': self.parse_errors,
        }
