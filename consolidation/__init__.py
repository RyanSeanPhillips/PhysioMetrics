"""
Consolidation Module - Manages multi-file data consolidation and Excel export.

This module extracts consolidation functionality from main.py for better
maintainability and easier customization for different experiment types.
"""

from .consolidation_manager import ConsolidationManager

__all__ = ['ConsolidationManager']
