"""
File Table Model - Model/View architecture for Project Builder table.

This module provides a QAbstractTableModel implementation that replaces
the hardcoded QTableWidget approach, enabling:
- Column reordering via drag-drop
- Column hiding/showing
- Custom user-defined columns
- Cleaner separation of data and presentation

Usage:
    from core.file_table_model import FileTableModel, ColumnDef

    model = FileTableModel()
    table_view.setModel(model)

    # Add data
    model.set_files(files_metadata_list)

    # Get column index by key (instead of hardcoding)
    exp_col = model.get_column_index('experiment')
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

from PyQt6.QtCore import (
    QAbstractTableModel, QModelIndex, Qt, QVariant,
    QSettings, pyqtSignal
)
from PyQt6.QtGui import QColor, QBrush, QIcon
from PyQt6.QtWidgets import QStyle, QApplication


class ColumnType(Enum):
    """Types of columns with different rendering/editing behavior."""
    TEXT = "text"           # Simple editable text
    TEXT_READONLY = "text_readonly"  # Non-editable text
    STATUS = "status"       # Status indicator with icon
    BUTTON = "button"       # Action buttons (handled by delegate)
    EXPORTS = "exports"     # Export status display
    COMBOBOX = "combobox"   # Dropdown selection


@dataclass
class ColumnDef:
    """Definition for a table column."""
    key: str                    # Internal identifier (e.g., 'experiment', 'strain')
    header: str                 # Display name (e.g., 'Experiment', 'Strain')
    width: int = 70             # Default width in pixels
    min_width: int = 40         # Minimum width
    column_type: ColumnType = ColumnType.TEXT_READONLY
    editable: bool = False      # Can user edit this column?
    expandable: bool = False    # Gets extra space when available?
    fixed: bool = False         # Fixed width (e.g., action buttons)?
    history_key: str = ""       # QSettings key for autocomplete history (optional)
    tooltip: str = ""           # Column header tooltip
    hidden: bool = False        # Is column hidden by default?


# Default column definitions - order here is the default display order
DEFAULT_COLUMNS: List[ColumnDef] = [
    ColumnDef(
        key='file_name',
        header='File Name',
        width=70,
        min_width=60,
        column_type=ColumnType.TEXT_READONLY,
        expandable=True,
        tooltip="ABF file name"
    ),
    ColumnDef(
        key='protocol',
        header='Protocol',
        width=58,
        min_width=40,
        column_type=ColumnType.TEXT_READONLY,
        expandable=True,
        tooltip="Recording protocol"
    ),
    ColumnDef(
        key='channel_count',
        header='Avail Ch',
        width=52,
        min_width=40,
        column_type=ColumnType.TEXT_READONLY,
        tooltip="Available channels in file"
    ),
    ColumnDef(
        key='sweep_count',
        header='Sweeps',
        width=50,
        min_width=40,
        column_type=ColumnType.TEXT_READONLY,
        tooltip="Number of sweeps"
    ),
    ColumnDef(
        key='keywords',
        header='Keywords',
        width=62,
        min_width=40,
        column_type=ColumnType.TEXT_READONLY,
        expandable=True,
        tooltip="Keywords extracted from file path"
    ),
    ColumnDef(
        key='experiment',
        header='Experiment',
        width=70,
        min_width=50,
        column_type=ColumnType.TEXT,
        editable=True,
        expandable=True,
        history_key='experiment_history',
        tooltip="User-defined experiment grouping"
    ),
    ColumnDef(
        key='channel',
        header='Channel',
        width=55,
        min_width=40,
        column_type=ColumnType.TEXT_READONLY,
        tooltip="Channel to analyze"
    ),
    ColumnDef(
        key='stim_channel',
        header='Stim Ch',
        width=52,
        min_width=40,
        column_type=ColumnType.TEXT_READONLY,
        tooltip="Stimulation channel"
    ),
    ColumnDef(
        key='events_channel',
        header='Events Ch',
        width=62,
        min_width=40,
        column_type=ColumnType.TEXT_READONLY,
        tooltip="Events/TTL channel"
    ),
    ColumnDef(
        key='strain',
        header='Strain',
        width=45,
        min_width=35,
        column_type=ColumnType.TEXT,
        editable=True,
        history_key='strain_history',
        tooltip="Mouse strain"
    ),
    ColumnDef(
        key='state',
        header='State',
        width=45,
        min_width=35,
        column_type=ColumnType.TEXT,
        editable=True,
        tooltip="Animal state (awake, iso, urethane, etc.)"
    ),
    ColumnDef(
        key='stim_type',
        header='Stim Type',
        width=62,
        min_width=40,
        column_type=ColumnType.TEXT,
        editable=True,
        tooltip="Stimulation type/parameters"
    ),
    ColumnDef(
        key='power',
        header='Power',
        width=45,
        min_width=30,
        column_type=ColumnType.TEXT,
        editable=True,
        tooltip="Laser/stim power"
    ),
    ColumnDef(
        key='sex',
        header='Sex',
        width=30,
        min_width=25,
        column_type=ColumnType.TEXT,
        editable=True,
        tooltip="Animal sex (M/F)"
    ),
    ColumnDef(
        key='animal_id',
        header='Animal ID',
        width=62,
        min_width=40,
        column_type=ColumnType.TEXT,
        editable=True,
        tooltip="Animal identifier"
    ),
    ColumnDef(
        key='status',
        header='Status',
        width=55,
        min_width=45,
        column_type=ColumnType.STATUS,
        tooltip="Analysis status"
    ),
    ColumnDef(
        key='actions',
        header='',
        width=85,
        min_width=85,
        column_type=ColumnType.BUTTON,
        fixed=True,
        tooltip="Actions"
    ),
    ColumnDef(
        key='exports',
        header='Exports',
        width=55,
        min_width=40,
        column_type=ColumnType.EXPORTS,
        expandable=True,
        tooltip="Export status and files"
    ),
    ColumnDef(
        key='linked_notes',
        header='Notes',
        width=45,
        min_width=35,
        column_type=ColumnType.TEXT_READONLY,
        tooltip="Notes files that reference this data file (click to preview)"
    ),
]


class FileTableModel(QAbstractTableModel):
    """
    Table model for Project Builder file list.

    Manages file metadata and provides a clean interface for the view.
    Columns are defined via ColumnDef objects rather than hardcoded indices.
    """

    # Signals
    data_changed = pyqtSignal()  # Emitted when any data changes
    cell_edited = pyqtSignal(int, str, object)  # row, column_key, new_value

    # Custom roles
    FilePathRole = Qt.ItemDataRole.UserRole + 1
    RowDataRole = Qt.ItemDataRole.UserRole + 2
    ColumnKeyRole = Qt.ItemDataRole.UserRole + 3
    ConflictRole = Qt.ItemDataRole.UserRole + 4  # For conflict highlighting

    # Verification status constants
    VERIFIED = "verified"           # source_link exists, confidence >= 0.8
    LOW_CONFIDENCE = "low"          # source_link exists, confidence < 0.8
    DISAGREEMENT = "disagreement"   # multiple source_links with different values

    # Background colors for verification status
    _VERIFIED_COLOR = QColor(60, 140, 100, 50)      # subtle green
    _LOW_CONF_COLOR = QColor(180, 160, 60, 45)      # subtle yellow
    _DISAGREE_COLOR = QColor(200, 120, 50, 55)       # subtle orange

    # Quality coloring for missing/unverified fields
    _MISSING_COLOR = QColor(200, 80, 80, 45)         # subtle red — empty required field
    _UNVERIFIED_COLOR = QColor(140, 140, 140, 30)    # subtle gray — has value, no source_link

    # Metadata fields that can be verified via source links
    VERIFIABLE_FIELDS = {'strain', 'sex', 'animal_id', 'stim_type', 'power', 'experiment', 'state'}

    def __init__(self, parent=None):
        super().__init__(parent)

        # Data storage
        self._rows: List[Dict[str, Any]] = []

        # Column configuration
        self._column_defs: List[ColumnDef] = list(DEFAULT_COLUMNS)
        self._column_order: List[str] = [col.key for col in DEFAULT_COLUMNS]
        self._hidden_columns: set = set()
        self._custom_columns: List[ColumnDef] = []

        # Build key -> index mapping
        self._rebuild_column_index()

        # Conflict tracking (row indices with conflicts)
        self._conflict_rows: set = set()

        # Verification cache: {(animal_id, field): status}
        self._verification_cache: Dict[tuple, str] = {}

        # Quality coloring toggle (red=missing, green=verified, etc.)
        self._quality_coloring: bool = True

        # Load saved column configuration
        self._load_column_config()

    def _rebuild_column_index(self):
        """Rebuild the key -> column index mapping."""
        self._key_to_index: Dict[str, int] = {}
        for i, key in enumerate(self._column_order):
            if key not in self._hidden_columns:
                self._key_to_index[key] = i

    def _load_column_config(self):
        """Load column configuration from QSettings."""
        settings = QSettings("PhysioMetrics", "BreathAnalysis")

        # Load column order
        saved_order = settings.value("project_builder/column_order", None)
        if saved_order and isinstance(saved_order, list):
            # Validate that all keys exist
            valid_keys = {col.key for col in self._column_defs}
            if all(k in valid_keys for k in saved_order):
                self._column_order = saved_order
                self._rebuild_column_index()

        # Load hidden columns
        hidden = settings.value("project_builder/hidden_columns", [])
        if hidden and isinstance(hidden, list):
            self._hidden_columns = set(hidden)

    def save_column_config(self):
        """Save column configuration to QSettings."""
        settings = QSettings("PhysioMetrics", "BreathAnalysis")
        settings.setValue("project_builder/column_order", self._column_order)
        settings.setValue("project_builder/hidden_columns", list(self._hidden_columns))

    # -------------------------------------------------------------------------
    # Required QAbstractTableModel methods
    # -------------------------------------------------------------------------

    def rowCount(self, parent=QModelIndex()) -> int:
        """Return number of rows."""
        if parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        """Return number of visible columns."""
        if parent.isValid():
            return 0
        return len(self._column_order) - len(self._hidden_columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        """Return data for the given index and role."""
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()

        if row < 0 or row >= len(self._rows):
            return None

        # Get column definition
        col_def = self._get_column_def_for_index(col)
        if col_def is None:
            return None

        row_data = self._rows[row]

        if role == Qt.ItemDataRole.DisplayRole:
            return self._get_display_value(row_data, col_def)

        elif role == Qt.ItemDataRole.EditRole:
            return row_data.get(col_def.key, "")

        elif role == Qt.ItemDataRole.ToolTipRole:
            value = row_data.get(col_def.key, "")
            # Special tooltip for fuzzy-matched notes
            if col_def.key == 'linked_notes':
                display_val = str(value) if value else ""
                if display_val.startswith('~'):
                    fuzzy_stems = row_data.get('linked_notes_fuzzy_stems', [])
                    if fuzzy_stems:
                        stems_str = ', '.join(fuzzy_stems[:5])
                        if len(fuzzy_stems) > 5:
                            stems_str += f" +{len(fuzzy_stems) - 5} more"
                        return f"⚠ Fuzzy match: Notes reference nearby file(s): {stems_str}"
                    return "⚠ Fuzzy match: No exact match found, showing notes for nearby files"
            # Verification tooltip for metadata fields
            if col_def.key in self.VERIFIABLE_FIELDS:
                animal_id = row_data.get('animal_id', '')
                if animal_id:
                    status = self._verification_cache.get((animal_id, col_def.key))
                    if status == self.VERIFIED:
                        return f"{col_def.header}: verified from source document"
                    elif status == self.LOW_CONFIDENCE:
                        return f"{col_def.header}: extracted (low confidence)"
                    elif status == self.DISAGREEMENT:
                        return f"{col_def.header}: conflicting values in sources"
            if value and len(str(value)) > 20:
                return str(value)
            return None

        elif role == Qt.ItemDataRole.BackgroundRole:
            # Highlight conflicts (whole row)
            if row in self._conflict_rows:
                return QBrush(QColor(100, 80, 50))  # Brownish highlight
            # Per-cell quality coloring for metadata fields
            if self._quality_coloring and col_def.key in self.VERIFIABLE_FIELDS:
                value = row_data.get(col_def.key, '')
                # Empty required field → red
                if not value or (isinstance(value, str) and not value.strip()):
                    return QBrush(self._MISSING_COLOR)
                # Has value — check verification status
                animal_id = row_data.get('animal_id', '')
                if animal_id:
                    status = self._verification_cache.get((animal_id, col_def.key))
                    if status == self.VERIFIED:
                        return QBrush(self._VERIFIED_COLOR)
                    elif status == self.LOW_CONFIDENCE:
                        return QBrush(self._LOW_CONF_COLOR)
                    elif status == self.DISAGREEMENT:
                        return QBrush(self._DISAGREE_COLOR)
                # Has value but no source_link → subtle gray
                return QBrush(self._UNVERIFIED_COLOR)
            return None

        elif role == Qt.ItemDataRole.ForegroundRole:
            # Status column coloring
            if col_def.key == 'status':
                status = row_data.get('status', '')
                if status == 'completed':
                    return QBrush(QColor(100, 200, 100))  # Green
                elif status == 'error':
                    return QBrush(QColor(200, 100, 100))  # Red
                elif status == 'in_progress':
                    return QBrush(QColor(200, 200, 100))  # Yellow
            return None

        elif role == self.FilePathRole:
            return row_data.get('file_path', '')

        elif role == self.RowDataRole:
            return row_data

        elif role == self.ColumnKeyRole:
            return col_def.key

        elif role == self.ConflictRole:
            return row in self._conflict_rows

        return None

    def _get_display_value(self, row_data: Dict, col_def: ColumnDef) -> str:
        """Get the display string for a cell."""
        value = row_data.get(col_def.key, "")

        if col_def.key == 'file_name':
            return str(value) if value else ""

        elif col_def.key == 'channel_count':
            # Format channel count
            return str(value) if value else ""

        elif col_def.key == 'sweep_count':
            return str(value) if value else ""

        elif col_def.key == 'keywords':
            # Keywords column uses 'keywords_display' in the data
            display_val = row_data.get('keywords_display', '')
            return str(display_val) if display_val else ""

        elif col_def.key == 'status':
            # Return raw status value - StatusDelegate handles the icon conversion
            return str(value) if value else 'pending'

        elif col_def.key == 'exports':
            # Format exports display
            if isinstance(value, dict):
                # Format dict exports like "2 CSV, 1 NPZ, Session"
                parts = []
                csv_count = sum([
                    1 if value.get('timeseries_csv') else 0,
                    1 if value.get('breaths_csv') else 0,
                    1 if value.get('events_csv') else 0,
                ])
                if value.get('pdf'):
                    parts.append('1 PDF')
                if csv_count > 0:
                    parts.append(f'{csv_count} CSV')
                if value.get('npz'):
                    parts.append('1 NPZ')
                if value.get('ml_training'):
                    parts.append('ML')
                if value.get('session_state'):
                    parts.append('Session')
                return ', '.join(parts) if parts else ''
            elif isinstance(value, list):
                return ', '.join(str(v) for v in value)
            return str(value) if value else ""

        return str(value) if value is not None else ""

    def setData(self, index: QModelIndex, value, role: int = Qt.ItemDataRole.EditRole) -> bool:
        """Set data at the given index."""
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False

        row = index.row()
        col = index.column()

        if row < 0 or row >= len(self._rows):
            return False

        col_def = self._get_column_def_for_index(col)
        if col_def is None or not col_def.editable:
            return False

        # Update the data
        old_value = self._rows[row].get(col_def.key)
        self._rows[row][col_def.key] = value

        # Emit signals
        self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole])
        self.cell_edited.emit(row, col_def.key, value)

        return True

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        """Return header data."""
        if orientation == Qt.Orientation.Horizontal:
            if role == Qt.ItemDataRole.DisplayRole:
                col_def = self._get_column_def_for_index(section)
                return col_def.header if col_def else ""

            elif role == Qt.ItemDataRole.ToolTipRole:
                col_def = self._get_column_def_for_index(section)
                return col_def.tooltip if col_def else ""

        elif orientation == Qt.Orientation.Vertical:
            if role == Qt.ItemDataRole.DisplayRole:
                return str(section + 1)

        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        """Return item flags for the given index."""
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags

        base_flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

        col_def = self._get_column_def_for_index(index.column())
        if col_def and col_def.editable:
            base_flags |= Qt.ItemFlag.ItemIsEditable

        return base_flags

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _get_column_def_for_index(self, col_index: int) -> Optional[ColumnDef]:
        """Get the ColumnDef for a visual column index."""
        visible_cols = [k for k in self._column_order if k not in self._hidden_columns]
        if 0 <= col_index < len(visible_cols):
            key = visible_cols[col_index]
            for col_def in self._column_defs:
                if col_def.key == key:
                    return col_def
            # Check custom columns
            for col_def in self._custom_columns:
                if col_def.key == key:
                    return col_def
        return None

    def get_column_def(self, key: str) -> Optional[ColumnDef]:
        """Get ColumnDef by key."""
        for col_def in self._column_defs:
            if col_def.key == key:
                return col_def
        for col_def in self._custom_columns:
            if col_def.key == key:
                return col_def
        return None

    def get_column_index(self, key: str) -> int:
        """
        Get the visual column index for a column key.

        Returns -1 if column not found or hidden.
        """
        visible_cols = [k for k in self._column_order if k not in self._hidden_columns]
        try:
            return visible_cols.index(key)
        except ValueError:
            return -1

    def get_column_key(self, col_index: int) -> Optional[str]:
        """Get the column key for a visual column index."""
        visible_cols = [k for k in self._column_order if k not in self._hidden_columns]
        if 0 <= col_index < len(visible_cols):
            return visible_cols[col_index]
        return None

    def get_visible_columns(self) -> List[ColumnDef]:
        """Get list of visible column definitions in order."""
        result = []
        for key in self._column_order:
            if key not in self._hidden_columns:
                col_def = self.get_column_def(key)
                if col_def:
                    result.append(col_def)
        return result

    def get_all_column_defs(self) -> List[ColumnDef]:
        """Get all column definitions (including hidden and custom)."""
        return self._column_defs + self._custom_columns

    # -------------------------------------------------------------------------
    # Data management
    # -------------------------------------------------------------------------

    def set_files(self, files: List[Dict[str, Any]]):
        """
        Set the file data for the table.

        Args:
            files: List of dicts with file metadata
        """
        self.beginResetModel()
        self._rows = [dict(f) for f in files]  # Copy the data
        self._conflict_rows.clear()
        self.endResetModel()
        self.data_changed.emit()

    def add_file(self, file_data: Dict[str, Any]):
        """Add a single file to the table."""
        row = len(self._rows)
        self.beginInsertRows(QModelIndex(), row, row)
        self._rows.append(dict(file_data))
        self.endInsertRows()
        self.data_changed.emit()

    def remove_row(self, row: int) -> bool:
        """Remove a row from the table."""
        if 0 <= row < len(self._rows):
            self.beginRemoveRows(QModelIndex(), row, row)
            del self._rows[row]
            # Update conflict rows
            self._conflict_rows = {r - 1 if r > row else r for r in self._conflict_rows if r != row}
            self.endRemoveRows()
            self.data_changed.emit()
            return True
        return False

    def clear(self):
        """Clear all data from the table."""
        self.beginResetModel()
        self._rows.clear()
        self._conflict_rows.clear()
        self.endResetModel()
        self.data_changed.emit()

    def get_row_data(self, row: int) -> Optional[Dict[str, Any]]:
        """Get the data dict for a row."""
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None

    def get_all_data(self) -> List[Dict[str, Any]]:
        """Get all row data as a list of dicts."""
        return [dict(row) for row in self._rows]

    def update_row(self, row: int, data: Dict[str, Any]):
        """Update a row with new data."""
        if 0 <= row < len(self._rows):
            self._rows[row].update(data)
            # Emit change for all columns
            left = self.index(row, 0)
            right = self.index(row, self.columnCount() - 1)
            self.dataChanged.emit(left, right)
            self.data_changed.emit()

    def set_cell_value(self, row: int, key: str, value: Any):
        """Set a cell value by row and column key."""
        if 0 <= row < len(self._rows):
            self._rows[row][key] = value
            col_index = self.get_column_index(key)
            if col_index >= 0:
                index = self.index(row, col_index)
                self.dataChanged.emit(index, index)
            self.data_changed.emit()

    def get_cell_value(self, row: int, key: str) -> Any:
        """Get a cell value by row and column key."""
        if 0 <= row < len(self._rows):
            return self._rows[row].get(key)
        return None

    # -------------------------------------------------------------------------
    # Conflict management
    # -------------------------------------------------------------------------

    def set_conflict_rows(self, rows: set):
        """Set which rows have conflicts (for highlighting)."""
        old_conflicts = self._conflict_rows
        self._conflict_rows = set(rows)

        # Emit change for affected rows
        all_affected = old_conflicts | self._conflict_rows
        for row in all_affected:
            if 0 <= row < len(self._rows):
                left = self.index(row, 0)
                right = self.index(row, self.columnCount() - 1)
                self.dataChanged.emit(left, right, [Qt.ItemDataRole.BackgroundRole])

    def add_conflict_row(self, row: int):
        """Mark a row as having a conflict."""
        if row not in self._conflict_rows:
            self._conflict_rows.add(row)
            left = self.index(row, 0)
            right = self.index(row, self.columnCount() - 1)
            self.dataChanged.emit(left, right, [Qt.ItemDataRole.BackgroundRole])

    def clear_conflict_row(self, row: int):
        """Clear conflict status for a row."""
        if row in self._conflict_rows:
            self._conflict_rows.discard(row)
            left = self.index(row, 0)
            right = self.index(row, self.columnCount() - 1)
            self.dataChanged.emit(left, right, [Qt.ItemDataRole.BackgroundRole])

    # -------------------------------------------------------------------------
    # Verification status (source link color coding)
    # -------------------------------------------------------------------------

    def refresh_verification_cache(self, source_links: List[Dict[str, Any]]):
        """Rebuild the verification cache from source links.

        Call this after project load or source link updates.

        Args:
            source_links: All source links (from ProjectService.get_source_links()).
        """
        self._verification_cache.clear()

        # Group links by (animal_id, field)
        grouped: Dict[tuple, List[Dict]] = {}
        for link in source_links:
            key = (link.get("animal_id", ""), link.get("field", ""))
            if key[0] and key[1]:
                grouped.setdefault(key, []).append(link)

        for (animal_id, field), links in grouped.items():
            if field not in self.VERIFIABLE_FIELDS:
                continue

            # Check for disagreements
            values = set(str(l.get("value", "")).strip().lower() for l in links)
            if len(values) > 1:
                self._verification_cache[(animal_id, field)] = self.DISAGREEMENT
            else:
                # Check confidence
                max_conf = max((l.get("confidence", 0) or 0) for l in links)
                if max_conf >= 0.8:
                    self._verification_cache[(animal_id, field)] = self.VERIFIED
                else:
                    self._verification_cache[(animal_id, field)] = self.LOW_CONFIDENCE

        # Notify view to repaint
        if self._rows:
            top_left = self.index(0, 0)
            bottom_right = self.index(len(self._rows) - 1, self.columnCount() - 1)
            self.dataChanged.emit(top_left, bottom_right, [Qt.ItemDataRole.BackgroundRole])

    def clear_verification_cache(self):
        """Clear all verification status (e.g. when project changes)."""
        self._verification_cache.clear()
        if self._rows:
            top_left = self.index(0, 0)
            bottom_right = self.index(len(self._rows) - 1, self.columnCount() - 1)
            self.dataChanged.emit(top_left, bottom_right, [Qt.ItemDataRole.BackgroundRole])

    def set_quality_coloring(self, enabled: bool):
        """Toggle metadata quality coloring (red=missing, green=verified, etc.)."""
        if self._quality_coloring == enabled:
            return
        self._quality_coloring = enabled
        if self._rows:
            top_left = self.index(0, 0)
            bottom_right = self.index(len(self._rows) - 1, self.columnCount() - 1)
            self.dataChanged.emit(top_left, bottom_right, [Qt.ItemDataRole.BackgroundRole])

    # -------------------------------------------------------------------------
    # Column management
    # -------------------------------------------------------------------------

    def hide_column(self, key: str):
        """Hide a column."""
        if key not in self._hidden_columns:
            # Find the visual index before hiding
            col_index = self.get_column_index(key)
            if col_index >= 0:
                self.beginRemoveColumns(QModelIndex(), col_index, col_index)
                self._hidden_columns.add(key)
                self.endRemoveColumns()
                self.save_column_config()

    def show_column(self, key: str):
        """Show a hidden column."""
        if key in self._hidden_columns:
            self._hidden_columns.discard(key)
            # Find where to insert
            col_index = self.get_column_index(key)
            self.beginInsertColumns(QModelIndex(), col_index, col_index)
            self.endInsertColumns()
            self.save_column_config()

    def show_all_columns(self):
        """Show all hidden columns."""
        if self._hidden_columns:
            self.beginResetModel()
            self._hidden_columns.clear()
            self.endResetModel()
            self.save_column_config()

    def move_column(self, from_key: str, to_key: str):
        """Move a column to a new position (before to_key)."""
        if from_key not in self._column_order or to_key not in self._column_order:
            return

        from_idx = self._column_order.index(from_key)
        to_idx = self._column_order.index(to_key)

        self._column_order.remove(from_key)
        # Adjust to_idx if necessary
        if from_idx < to_idx:
            to_idx -= 1
        self._column_order.insert(to_idx, from_key)

        self._rebuild_column_index()
        self.save_column_config()

        # Notify view of column change
        self.layoutChanged.emit()

    def reorder_columns(self, new_order: List[str]):
        """Set a new column order."""
        # Validate
        valid_keys = {col.key for col in self._column_defs + self._custom_columns}
        if not all(k in valid_keys for k in new_order):
            return

        self.beginResetModel()
        self._column_order = new_order
        self._rebuild_column_index()
        self.endResetModel()
        self.save_column_config()

    def reset_column_order(self):
        """Reset column order to default."""
        self.beginResetModel()
        self._column_order = [col.key for col in DEFAULT_COLUMNS]
        self._hidden_columns.clear()
        self._rebuild_column_index()
        self.endResetModel()
        self.save_column_config()

    # -------------------------------------------------------------------------
    # Custom columns
    # -------------------------------------------------------------------------

    def add_custom_column(self, name: str, default_value: str = "") -> str:
        """
        Add a custom user-defined column.

        Args:
            name: Display name for the column
            default_value: Default value for existing rows

        Returns:
            The key for the new column
        """
        # Generate unique key
        key = f"custom_{name.lower().replace(' ', '_')}"
        counter = 1
        while any(c.key == key for c in self._column_defs + self._custom_columns):
            key = f"custom_{name.lower().replace(' ', '_')}_{counter}"
            counter += 1

        # Create column definition
        col_def = ColumnDef(
            key=key,
            header=name,
            width=70,
            min_width=40,
            column_type=ColumnType.TEXT,
            editable=True,
            tooltip=f"Custom column: {name}"
        )

        # Add to list and order
        self._custom_columns.append(col_def)
        self._column_order.append(key)
        self._rebuild_column_index()

        # Add default value to existing rows
        for row in self._rows:
            row[key] = default_value

        # Notify view
        col_index = self.get_column_index(key)
        self.beginInsertColumns(QModelIndex(), col_index, col_index)
        self.endInsertColumns()

        self.save_column_config()
        return key

    def remove_custom_column(self, key: str) -> bool:
        """Remove a custom column."""
        # Find the custom column
        col_def = None
        for c in self._custom_columns:
            if c.key == key:
                col_def = c
                break

        if col_def is None:
            return False

        col_index = self.get_column_index(key)
        if col_index < 0:
            return False

        self.beginRemoveColumns(QModelIndex(), col_index, col_index)

        # Remove from lists
        self._custom_columns.remove(col_def)
        self._column_order.remove(key)
        self._hidden_columns.discard(key)
        self._rebuild_column_index()

        # Remove data from rows
        for row in self._rows:
            row.pop(key, None)

        self.endRemoveColumns()
        self.save_column_config()
        return True

    def get_custom_columns(self) -> List[ColumnDef]:
        """Get list of custom column definitions."""
        return list(self._custom_columns)

    def sync_custom_columns_from_db(self, db_columns: List[Dict[str, Any]]):
        """
        Sync custom column definitions from the experiment store.

        Adds any DB-defined custom columns that aren't already in the model.
        Called when a project is loaded to pick up dynamic columns.

        Args:
            db_columns: List of dicts with keys: column_key, display_name, column_type
                        (from ExperimentStore.get_dynamic_columns() or PRAGMA table_info)
        """
        existing_keys = {c.key for c in self._column_defs + self._custom_columns}

        for db_col in db_columns:
            key = db_col.get("column_key", "")
            if not key or key in existing_keys:
                continue

            display_name = db_col.get("display_name", key)

            col_def = ColumnDef(
                key=key,
                header=display_name,
                width=70,
                min_width=40,
                column_type=ColumnType.TEXT,
                editable=True,
                tooltip=f"Custom column: {display_name}"
            )

            self._custom_columns.append(col_def)
            self._column_order.append(key)
            existing_keys.add(key)

        self._rebuild_column_index()
        self.beginResetModel()
        self.endResetModel()


# Convenience function to get default column keys
def get_default_column_keys() -> List[str]:
    """Get list of default column keys."""
    return [col.key for col in DEFAULT_COLUMNS]


def get_editable_column_keys() -> List[str]:
    """Get list of editable column keys."""
    return [col.key for col in DEFAULT_COLUMNS if col.editable]
