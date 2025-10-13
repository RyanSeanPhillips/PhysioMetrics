"""
Export Module - Manages data export and summary generation.

This module extracts export functionality from main.py for better maintainability
and easier customization for different experiment types.
"""

from .export_manager import ExportManager

__all__ = ['ExportManager']
