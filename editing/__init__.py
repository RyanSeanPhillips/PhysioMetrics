"""
Editing modes module for PhysioMetrics.

Provides interactive editing capabilities including:
- Peak addition and deletion
- Sigh marking
- Point movement (drag peaks/onsets/offsets)
- Sniffing bout region marking
"""

from .editing_modes import EditingModes

__all__ = ['EditingModes']
