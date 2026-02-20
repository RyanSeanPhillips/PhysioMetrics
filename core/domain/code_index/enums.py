"""
Code index enums.

Severity levels and access types for static analysis.
"""

from enum import Enum


class Severity(Enum):
    """Diagnostic severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class AccessType(Enum):
    """Attribute access types."""
    READ = "read"
    WRITE = "write"
    CALL = "call"
    CONNECT = "connect"
