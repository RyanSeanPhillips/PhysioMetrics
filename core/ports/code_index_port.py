"""
Code index database port (abstract interface).

Defines the contract for code index storage backends.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from core.domain.code_index.models import (
    FileDef, ImportDef, ClassDef, FunctionDef, AttributeAccess,
    SignalDef, ConnectionDef, CallDef, UIWidget, Diagnostic,
    KnowledgeEntry, IndexStats,
)


class CodeIndexPort(ABC):
    """Abstract interface for code index storage."""

    # --- File operations ---

    @abstractmethod
    def upsert_file(self, f: FileDef) -> FileDef:
        """Insert or update a file record. Returns the record with file_id set."""
        ...

    @abstractmethod
    def get_file(self, file_id: int) -> Optional[FileDef]:
        ...

    @abstractmethod
    def get_file_by_path(self, rel_path: str) -> Optional[FileDef]:
        ...

    @abstractmethod
    def list_files(self) -> List[FileDef]:
        ...

    @abstractmethod
    def delete_file(self, file_id: int) -> bool:
        """Delete file and all dependent rows (CASCADE)."""
        ...

    # --- Bulk insert (called after parsing a file) ---

    @abstractmethod
    def bulk_insert_imports(self, file_id: int, imports: List[ImportDef]) -> None:
        ...

    @abstractmethod
    def bulk_insert_classes(self, file_id: int, classes: List[ClassDef]) -> List[ClassDef]:
        """Insert classes and return them with class_id set (needed for functions)."""
        ...

    @abstractmethod
    def bulk_insert_functions(self, file_id: int, functions: List[FunctionDef]) -> List[FunctionDef]:
        """Insert functions and return them with func_id set (needed for calls/attrs)."""
        ...

    @abstractmethod
    def bulk_insert_attributes(self, file_id: int, attrs: List[AttributeAccess]) -> None:
        ...

    @abstractmethod
    def bulk_insert_signals(self, file_id: int, signals: List[SignalDef]) -> None:
        ...

    @abstractmethod
    def bulk_insert_connections(self, file_id: int, connections: List[ConnectionDef]) -> None:
        ...

    @abstractmethod
    def bulk_insert_calls(self, file_id: int, calls: List[CallDef]) -> None:
        ...

    @abstractmethod
    def bulk_insert_ui_widgets(self, ui_file: str, widgets: List[UIWidget]) -> None:
        ...

    @abstractmethod
    def bulk_insert_diagnostics(self, diagnostics: List[Diagnostic]) -> None:
        ...

    # --- Query operations ---

    @abstractmethod
    def find_functions(self, name: Optional[str] = None, class_name: Optional[str] = None,
                       file_pattern: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Find functions matching criteria. Returns dicts with file/class context."""
        ...

    @abstractmethod
    def find_classes(self, name: Optional[str] = None, base_class: Optional[str] = None,
                     limit: int = 50) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def get_callers(self, function_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Find all call sites that reference function_name."""
        ...

    @abstractmethod
    def get_callees(self, func_id: int) -> List[Dict[str, Any]]:
        """Find all calls made from a specific function."""
        ...

    @abstractmethod
    def get_signals(self, name: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Find signal declarations and their connections."""
        ...

    @abstractmethod
    def get_state_fields(self) -> List[Dict[str, Any]]:
        """Get AppState field accesses grouped by field name."""
        ...

    @abstractmethod
    def get_widget_refs(self, widget_name: str) -> List[Dict[str, Any]]:
        """Find all references to a UI widget."""
        ...

    @abstractmethod
    def get_diagnostics(self, severity: Optional[str] = None, rule_id: Optional[str] = None,
                        file_pattern: Optional[str] = None, limit: int = 100) -> List[Diagnostic]:
        ...

    @abstractmethod
    def search_fts(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Full-text search across function names, class names, docstrings."""
        ...

    @abstractmethod
    def get_file_summary(self, rel_path: str) -> Optional[Dict[str, Any]]:
        """Get structured summary of a file: classes, functions, imports, diagnostics."""
        ...

    # --- Knowledge cache ---

    @abstractmethod
    def set_knowledge(self, key: str, value: Any) -> None:
        ...

    @abstractmethod
    def get_knowledge(self, key: str) -> Optional[KnowledgeEntry]:
        ...

    @abstractmethod
    def delete_knowledge(self, key: str) -> bool:
        ...

    # --- Stats & maintenance ---

    @abstractmethod
    def get_stats(self) -> IndexStats:
        ...

    @abstractmethod
    def clear_diagnostics(self) -> None:
        """Remove all diagnostics (before re-running analysis)."""
        ...

    @abstractmethod
    def clear_ui_widgets(self) -> None:
        """Remove all UI widget records (before re-parsing .ui file)."""
        ...

    @abstractmethod
    def close(self) -> None:
        ...
