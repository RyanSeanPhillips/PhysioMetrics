"""
Channel Manager Widget

A collapsible widget for managing multiple data channels.
Allows users to:
- Select which channels to display
- Set channel types (Pleth, Photometry, Stim/TTL, etc.)
- Choose a primary channel for analysis
- Access settings for computed channels (like ΔF/F)
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QComboBox, QFrame, QSizePolicy,
    QSpacerItem, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QSize, QTimer
from PyQt6.QtGui import QFont, QCursor


# Channel type options (simplified for now)
CHANNEL_TYPES = [
    ("Pleth", "Plethysmography - peak detection and breath analysis"),
    ("Opto Stim", "Optogenetic stimulus - shows blue background during stim"),
    ("Raw Signal", "Display only, no special processing"),
]


@dataclass
class ChannelConfig:
    """Configuration for a single channel."""
    name: str
    visible: bool = False
    channel_type: str = "Raw Signal"
    source: str = "file"  # "file" or "computed"
    order: int = 0
    settings_callback: Optional[Callable] = None  # For computed channels

    def __post_init__(self):
        # Auto-detect channel type from name
        name_lower = self.name.lower()
        if 'pleth' in name_lower or 'resp' in name_lower or 'breath' in name_lower:
            self.channel_type = "Pleth"
        elif 'opto' in name_lower or 'laser' in name_lower or 'stim' in name_lower:
            self.channel_type = "Opto Stim"
            # Opto Stim channels default to hidden (still creates blue spans on Pleth)
            self.visible = False
        elif 'δf/f' in name_lower or 'dff' in name_lower or 'df/f' in name_lower:
            self.source = "computed"
            # Photometry is Raw Signal for display purposes
            self.channel_type = "Raw Signal"


class ChannelRowWidget(QFrame):
    """A single row in the channel list."""

    visibility_changed = pyqtSignal(str, bool)  # channel_name, is_visible
    type_changed = pyqtSignal(str, str)  # channel_name, new_type
    settings_requested = pyqtSignal(str)  # channel_name
    move_requested = pyqtSignal(str, int)  # channel_name, direction (-1=up, 1=down)

    def __init__(self, config: ChannelConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._setup_ui()

    def _setup_ui(self):
        # Subtle row background on hover
        self.setStyleSheet("""
            ChannelRowWidget {
                background-color: transparent;
                border-radius: 2px;
            }
            ChannelRowWidget:hover {
                background-color: #3a3a3a;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(2)

        # Move up/down buttons for reordering
        move_btn_style = """
            QPushButton {
                background-color: transparent;
                color: #666666;
                border: none;
                font-size: 8px;
                padding: 0px;
                min-width: 10px;
                max-width: 10px;
            }
            QPushButton:hover {
                color: #aaaaaa;
            }
            QPushButton:pressed {
                color: #ffffff;
            }
        """
        self.move_up_btn = QPushButton("▲")
        self.move_up_btn.setStyleSheet(move_btn_style)
        self.move_up_btn.setFixedSize(10, 10)
        self.move_up_btn.setToolTip("Move up")
        self.move_up_btn.clicked.connect(lambda: self.move_requested.emit(self.config.name, -1))
        layout.addWidget(self.move_up_btn)

        self.move_down_btn = QPushButton("▼")
        self.move_down_btn.setStyleSheet(move_btn_style)
        self.move_down_btn.setFixedSize(10, 10)
        self.move_down_btn.setToolTip("Move down")
        self.move_down_btn.clicked.connect(lambda: self.move_requested.emit(self.config.name, 1))
        layout.addWidget(self.move_down_btn)

        # Visibility checkbox
        self.visible_check = QCheckBox()
        self.visible_check.setChecked(self.config.visible)
        self.visible_check.setStyleSheet("""
            QCheckBox::indicator {
                width: 10px;
                height: 10px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #555555;
                border-radius: 2px;
                background: transparent;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #007acc;
                border-radius: 2px;
                background: #007acc;
            }
        """)
        self.visible_check.setToolTip("Show/hide this channel")
        self.visible_check.toggled.connect(self._on_visibility_toggled)
        layout.addWidget(self.visible_check)

        # Channel name - fixed width to align dropdowns
        self.name_label = QLabel(self.config.name)
        self.name_label.setStyleSheet("color: #cccccc; font-size: 9px;")
        self.name_label.setFixedWidth(65)
        layout.addWidget(self.name_label)

        # Channel type dropdown - compact
        self.type_combo = QComboBox()
        self.type_combo.setStyleSheet("""
            QComboBox {
                background-color: #3c3c3c;
                color: #aaaaaa;
                border: 1px solid #4a4a4a;
                border-radius: 2px;
                padding: 1px 2px;
                min-width: 65px;
                max-width: 70px;
                font-size: 9px;
            }
            QComboBox:hover {
                border-color: #5a5a5a;
            }
            QComboBox::drop-down {
                border: none;
                width: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d30;
                color: #cccccc;
                selection-background-color: #094771;
                font-size: 9px;
            }
        """)
        for type_name, type_desc in CHANNEL_TYPES:
            self.type_combo.addItem(type_name)
            self.type_combo.setItemData(self.type_combo.count() - 1, type_desc, Qt.ItemDataRole.ToolTipRole)

        # Set current type
        idx = self.type_combo.findText(self.config.channel_type)
        if idx >= 0:
            self.type_combo.setCurrentIndex(idx)
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        layout.addWidget(self.type_combo)

        # Settings button container - fixed width to maintain alignment
        self.settings_container = QWidget()
        self.settings_container.setFixedWidth(12)
        settings_layout = QHBoxLayout(self.settings_container)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(0)

        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setFixedSize(12, 12)
        self.settings_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #007acc;
                border: none;
                font-size: 9px;
                padding: 0px;
            }
            QPushButton:hover {
                color: #3399ff;
            }
        """)
        self.settings_btn.setToolTip("Open settings for this channel")
        self.settings_btn.clicked.connect(self._on_settings_clicked)
        self.settings_btn.setVisible(self.config.source == "computed")
        settings_layout.addWidget(self.settings_btn)

        layout.addWidget(self.settings_container)

    def _on_visibility_toggled(self, checked: bool):
        self.config.visible = checked
        self.visibility_changed.emit(self.config.name, checked)

    def _on_type_changed(self, new_type: str):
        self.config.channel_type = new_type
        self.type_changed.emit(self.config.name, new_type)

    def _on_settings_clicked(self):
        self.settings_requested.emit(self.config.name)

    def update_config(self, config: ChannelConfig):
        """Update the row with new config."""
        self.config = config
        self.visible_check.setChecked(config.visible)
        idx = self.type_combo.findText(config.channel_type)
        if idx >= 0:
            self.type_combo.setCurrentIndex(idx)
        # Show gear only for computed channels (container maintains alignment)
        self.settings_btn.setVisible(config.source == "computed")


class ChannelManagerWidget(QWidget):
    """
    Collapsible channel manager widget.

    Signals:
        channels_changed: Emitted when any channel configuration changes
        apply_requested: Emitted when user clicks Apply

    Args:
        parent: Parent widget
        summary_label: Optional external QLabel for summary text (from .ui file)
        expand_btn: Optional external QPushButton for expand/collapse (from .ui file)
        preview_label: Optional external QLabel for channel preview (from .ui file)
    """

    channels_changed = pyqtSignal()
    apply_requested = pyqtSignal()
    settings_requested = pyqtSignal(str)  # channel_name

    def __init__(self, parent=None, summary_label=None, expand_btn=None, preview_label=None):
        super().__init__(parent)
        self._is_expanded = False
        self._channels: Dict[str, ChannelConfig] = {}
        self._channel_rows: Dict[str, ChannelRowWidget] = {}
        self._just_closed = False  # Track if popup just closed to prevent re-open

        # Store external UI elements if provided
        self._external_summary_label = summary_label
        self._external_expand_btn = expand_btn
        self._external_preview_label = preview_label

        self._setup_ui()

    def _setup_ui(self):
        # No border - designed to fit inside a QGroupBox
        self.setStyleSheet("""
            ChannelManagerWidget {
                background-color: transparent;
            }
        """)

        # Check if we're using external UI elements (from .ui file)
        using_external_ui = (self._external_summary_label is not None or
                            self._external_expand_btn is not None or
                            self._external_preview_label is not None)

        if using_external_ui:
            # Use external UI elements - they're already in the .ui file layout
            self.summary_label = self._external_summary_label
            self.expand_btn = self._external_expand_btn
            self.preview_label = self._external_preview_label

            # Wire up expand button
            if self.expand_btn:
                self.expand_btn.clicked.connect(self.toggle_expanded)

            # For positioning the popup, use the summary label's parent (the groupbox layout area)
            # This ensures popup appears below the channel manager header, not the main window
            if self.summary_label and self.summary_label.parent():
                self.header = self.summary_label.parent()
            else:
                self.header = self

        else:
            # Create internal UI elements (standalone mode)
            main_layout = QVBoxLayout(self)
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.setSpacing(0)

            # Header (always visible) - clickable to expand
            self.header = QFrame()
            self.header.setStyleSheet("""
                QFrame {
                    background-color: transparent;
                }
            """)
            self.header.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            self.header.mousePressEvent = self._on_header_clicked

            header_layout = QVBoxLayout(self.header)
            header_layout.setContentsMargins(0, 0, 0, 4)
            header_layout.setSpacing(2)

            # Top row: summary + expand button
            top_row = QHBoxLayout()
            top_row.setSpacing(4)

            self.summary_label = QLabel("No channels loaded")
            self.summary_label.setStyleSheet("color: #cccccc; font-size: 10px;")
            top_row.addWidget(self.summary_label)

            top_row.addStretch()

            self.expand_btn = QPushButton("▼")
            self.expand_btn.setFixedSize(14, 14)
            self.expand_btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #666666;
                    border: none;
                    font-size: 8px;
                }
                QPushButton:hover {
                    color: #aaaaaa;
                }
            """)
            self.expand_btn.clicked.connect(self.toggle_expanded)
            top_row.addWidget(self.expand_btn)

            header_layout.addLayout(top_row)

            # Preview row: shows channel types when collapsed
            # Use fixed height to prevent layout shift when hidden
            self.preview_label = QLabel("")
            self.preview_label.setStyleSheet("color: #888888; font-size: 9px;")
            self.preview_label.setWordWrap(True)
            self.preview_label.setFixedHeight(14)  # Fixed height prevents layout shift
            header_layout.addWidget(self.preview_label)

            main_layout.addWidget(self.header)

        # Content area (popup-style dropdown - floats over other content)
        # Use Qt.Popup flag so it auto-closes when clicking outside
        self.content = QFrame(self, Qt.WindowType.Popup)
        self.content.setStyleSheet("""
            QFrame {
                background-color: #2d2d30;
                border: 1px solid #3e3e42;
                border-radius: 4px;
            }
        """)
        self.content.setVisible(False)

        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(2)

        # Channel list container - no scroll area, auto-sizes to fit all channels
        self.channel_list = QWidget()
        self.channel_list.setStyleSheet("background: transparent;")
        self.channel_list_layout = QVBoxLayout(self.channel_list)
        self.channel_list_layout.setContentsMargins(0, 0, 0, 0)
        self.channel_list_layout.setSpacing(2)
        self.channel_list_layout.addStretch()

        content_layout.addWidget(self.channel_list)

        # Button row at bottom of popup
        button_row = QHBoxLayout()
        button_row.setSpacing(4)

        # Show All toggle - compact checkbox style
        self.show_all_check = QCheckBox()
        self.show_all_check.setStyleSheet("""
            QCheckBox::indicator {
                width: 10px;
                height: 10px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #555555;
                border-radius: 2px;
                background: transparent;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #007acc;
                border-radius: 2px;
                background: #007acc;
            }
        """)
        self.show_all_check.setToolTip("Toggle all channels visible")
        self.show_all_check.clicked.connect(self._on_show_all_clicked)
        button_row.addWidget(self.show_all_check)

        show_all_label = QLabel("All")
        show_all_label.setStyleSheet("color: #999999; font-size: 9px;")
        button_row.addWidget(show_all_label)

        button_row.addStretch()

        # Apply button
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 6px 12px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
            QPushButton:pressed {
                background-color: #006cc1;
            }
        """)
        self.apply_btn.clicked.connect(self._on_apply_clicked)
        button_row.addWidget(self.apply_btn)

        content_layout.addLayout(button_row)

        # Note: content is NOT added to main_layout - it's a popup

        # Size constraints depend on mode
        if using_external_ui:
            # In external UI mode, the widget itself is not displayed
            # Only the popup content is shown - hide the widget
            self.setFixedSize(0, 0)
            self.hide()
        else:
            # Standalone mode - size is flexible
            self.setMinimumWidth(150)
            self.setMaximumWidth(200)

        # Popup size - minimum width only, let it expand to fit content
        self.content.setMinimumWidth(200)

        # Install event filter to detect when popup closes (clicking outside)
        self.content.installEventFilter(self)

    def eventFilter(self, obj, event):
        """Handle popup close events."""
        if obj == self.content and event.type() == event.Type.Hide:
            # Popup was closed (clicked outside or programmatically)
            if self._is_expanded:
                self._is_expanded = False
                self.expand_btn.setText("▼")
                # Restore preview text
                if hasattr(self, '_saved_preview'):
                    self.preview_label.setText(self._saved_preview)
                # Don't auto-apply when clicking outside - user must click Apply button
                # Set flag to prevent immediate re-open from header click
                self._just_closed = True
                QTimer.singleShot(200, self._clear_just_closed)
        return super().eventFilter(obj, event)

    def _on_apply_clicked(self):
        """Handle Apply button click - emit signal and close popup."""
        self.apply_requested.emit()
        # Close the popup
        self.content.hide()
        self._is_expanded = False
        self.expand_btn.setText("▼")
        if hasattr(self, '_saved_preview'):
            self.preview_label.setText(self._saved_preview)

    def _on_show_all_clicked(self):
        """Handle Show All checkbox click - toggle all channels visibility."""
        # If checkbox is checked, show all; if unchecked, hide all except first Pleth
        show_all = self.show_all_check.isChecked()

        first_pleth_found = False
        for name, config in self._channels.items():
            if show_all:
                config.visible = True
            else:
                # When unchecking, keep first Pleth visible, hide others
                if config.channel_type == "Pleth" and not first_pleth_found:
                    config.visible = True
                    first_pleth_found = True
                else:
                    config.visible = False

            # Update the UI checkbox
            if name in self._channel_rows:
                row = self._channel_rows[name]
                row.visible_check.blockSignals(True)
                row.visible_check.setChecked(config.visible)
                row.visible_check.blockSignals(False)

        self._update_summary()
        self._update_show_all_checkbox()
        self.channels_changed.emit()

    def _update_show_all_checkbox(self):
        """Update the Show All checkbox state based on current visibility."""
        if not hasattr(self, 'show_all_check'):
            return
        all_visible = all(c.visible for c in self._channels.values()) if self._channels else False
        self.show_all_check.blockSignals(True)
        self.show_all_check.setChecked(all_visible)
        self.show_all_check.blockSignals(False)

    def _clear_just_closed(self):
        """Clear the just_closed flag after a short delay."""
        self._just_closed = False

    def _on_header_clicked(self, event):
        # If popup just closed, don't re-open it immediately
        if self._just_closed:
            return
        self.toggle_expanded()

    def toggle_expanded(self):
        """Toggle between expanded and collapsed states."""
        was_expanded = self._is_expanded
        self._is_expanded = not self._is_expanded

        if self._is_expanded:
            # Resize popup to fit all content before showing
            self.content.adjustSize()
            # Position popup below the header
            global_pos = self.header.mapToGlobal(self.header.rect().bottomLeft())
            self.content.move(global_pos)
            self.content.show()
        else:
            self.content.hide()

        self.expand_btn.setText("▲" if self._is_expanded else "▼")
        # Clear preview when expanded (full list is visible) - don't hide to avoid layout shift
        if self._is_expanded:
            self._saved_preview = self.preview_label.text()
            self.preview_label.setText("")
        else:
            # Restore preview text when collapsed
            if hasattr(self, '_saved_preview'):
                self.preview_label.setText(self._saved_preview)
        # Don't auto-apply when collapsing - user must click Apply button

    def set_channels(self, channels):
        """
        Set the available channels.

        Args:
            channels: List of channel names (str) OR ChannelConfig objects
        """
        # Clear existing rows
        for row in self._channel_rows.values():
            row.setParent(None)
            row.deleteLater()
        self._channel_rows.clear()
        self._channels.clear()

        # Create configs for each channel
        first_pleth_found = False
        for i, item in enumerate(channels):
            # Accept either string names or ChannelConfig objects
            if isinstance(item, ChannelConfig):
                config = item
                config.order = i  # Ensure order is set
            else:
                # It's a string name - create config (triggers __post_init__ for type detection)
                config = ChannelConfig(
                    name=item,
                    order=i
                )
                # Set visibility based on channel type:
                # - First Pleth channel is visible by default
                # - Opto Stim channels are hidden by default
                # - Other channels: first one visible if no Pleth found yet
                if config.channel_type == "Pleth" and not first_pleth_found:
                    config.visible = True
                    first_pleth_found = True
                elif config.channel_type == "Opto Stim":
                    config.visible = False
                elif i == 0 and not first_pleth_found:
                    config.visible = True

            self._channels[config.name] = config

            # Create row widget
            row = ChannelRowWidget(config)
            row.visibility_changed.connect(self._on_channel_visibility_changed)
            row.type_changed.connect(self._on_channel_type_changed)
            row.settings_requested.connect(self._on_channel_settings_requested)
            row.move_requested.connect(self._on_channel_move_requested)

            self._channel_rows[config.name] = row
            # Insert before the stretch
            self.channel_list_layout.insertWidget(self.channel_list_layout.count() - 1, row)

        self._update_summary()

    def add_channel(self, name: str, channel_type: str = "Raw Signal",
                    source: str = "file", visible: bool = False,
                    settings_callback: Callable = None):
        """Add a single channel (e.g., computed ΔF/F)."""
        if name in self._channels:
            return  # Already exists

        config = ChannelConfig(
            name=name,
            visible=visible,
            channel_type=channel_type,
            source=source,
            order=len(self._channels),
            settings_callback=settings_callback
        )
        self._channels[name] = config

        row = ChannelRowWidget(config)
        row.visibility_changed.connect(self._on_channel_visibility_changed)
        row.type_changed.connect(self._on_channel_type_changed)
        row.settings_requested.connect(self._on_channel_settings_requested)
        row.move_requested.connect(self._on_channel_move_requested)

        self._channel_rows[name] = row
        self.channel_list_layout.insertWidget(self.channel_list_layout.count() - 1, row)

        self._update_summary()

    def remove_channel(self, name: str):
        """Remove a channel."""
        if name not in self._channels:
            return

        del self._channels[name]

        if name in self._channel_rows:
            row = self._channel_rows.pop(name)
            row.setParent(None)
            row.deleteLater()

        self._update_summary()

    def get_channels(self) -> Dict[str, 'ChannelConfig']:
        """Get all channel configurations.

        Returns:
            Dict mapping channel name to ChannelConfig
        """
        return self._channels.copy()

    def get_visible_channels(self) -> List[str]:
        """Get list of visible channel names in order."""
        visible = [(c.name, c.order) for c in self._channels.values() if c.visible]
        visible.sort(key=lambda x: x[1])
        return [name for name, _ in visible]

    def get_pleth_channel(self) -> Optional[str]:
        """Get the name of the Pleth channel (for analysis)."""
        for config in self._channels.values():
            if config.channel_type == "Pleth":
                return config.name
        return None

    def get_opto_stim_channel(self) -> Optional[str]:
        """Get the name of the Opto Stim channel (for blue overlay)."""
        for config in self._channels.values():
            if config.channel_type == "Opto Stim":
                return config.name
        return None

    def get_channel_type(self, name: str) -> Optional[str]:
        """Get the type of a channel."""
        if name in self._channels:
            return self._channels[name].channel_type
        return None

    def set_channel_type(self, name: str, channel_type: str):
        """Set the type of a channel and update the UI."""
        if name in self._channels:
            self._channels[name].channel_type = channel_type
            # Update the row widget if it exists
            if name in self._channel_rows:
                row = self._channel_rows[name]
                idx = row.type_combo.findText(channel_type)
                if idx >= 0:
                    row.type_combo.blockSignals(True)
                    row.type_combo.setCurrentIndex(idx)
                    row.type_combo.blockSignals(False)
            self._update_summary()

    def get_channels_by_type(self, channel_type: str) -> List[str]:
        """Get all channels of a specific type."""
        return [c.name for c in self._channels.values() if c.channel_type == channel_type]

    def get_all_configs(self) -> Dict[str, ChannelConfig]:
        """Get all channel configurations."""
        return self._channels.copy()

    def _update_summary(self):
        """Update the summary and preview labels in the header."""
        if not self._channels:
            self.summary_label.setText("No channels loaded")
            self.preview_label.setText("")
            return

        visible = self.get_visible_channels()
        visible_count = len(visible)
        total_count = len(self._channels)
        pleth = self.get_pleth_channel()
        opto = self.get_opto_stim_channel()

        # Summary line
        if visible_count == 0:
            self.summary_label.setText("No channels selected")
        elif visible_count == 1:
            self.summary_label.setText(f"{visible[0]} ({visible_count}/{total_count})")
        elif pleth and pleth in visible:
            self.summary_label.setText(f"{pleth} + {visible_count - 1} more ({visible_count}/{total_count})")
        else:
            self.summary_label.setText(f"{visible_count}/{total_count} channels")

        # Preview line - show visible channel names abbreviated
        if visible_count > 0:
            # Build preview showing types
            preview_parts = []
            for name in visible[:4]:  # Show max 4
                config = self._channels.get(name)
                if config:
                    # Abbreviate long names
                    short_name = name[:10] + "..." if len(name) > 12 else name
                    preview_parts.append(short_name)
            if visible_count > 4:
                preview_parts.append(f"+{visible_count - 4}")
            self.preview_label.setText(" · ".join(preview_parts))
        else:
            self.preview_label.setText("Click to configure channels")

    def _on_channel_visibility_changed(self, name: str, visible: bool):
        if name in self._channels:
            self._channels[name].visible = visible
        self._update_summary()
        self._update_show_all_checkbox()
        self.channels_changed.emit()

    def _on_channel_type_changed(self, name: str, new_type: str):
        if name in self._channels:
            self._channels[name].channel_type = new_type

            # Enforce single Pleth: if this channel is set to Pleth,
            # change all other Pleth channels to Raw Signal
            if new_type == "Pleth":
                for other_name, other_config in self._channels.items():
                    if other_name != name and other_config.channel_type == "Pleth":
                        # Update the config
                        other_config.channel_type = "Raw Signal"
                        # Update the UI row (block signals to prevent loop)
                        if other_name in self._channel_rows:
                            row = self._channel_rows[other_name]
                            row.type_combo.blockSignals(True)
                            idx = row.type_combo.findText("Raw Signal")
                            if idx >= 0:
                                row.type_combo.setCurrentIndex(idx)
                            row.type_combo.blockSignals(False)

        self.channels_changed.emit()

    def _on_channel_move_requested(self, name: str, direction: int):
        """Handle channel move request (up=-1, down=1)."""
        if name not in self._channels:
            return

        # Get sorted list of channels by order
        sorted_channels = sorted(self._channels.items(), key=lambda x: x[1].order)
        channel_names = [n for n, c in sorted_channels]

        # Find current index
        try:
            current_idx = channel_names.index(name)
        except ValueError:
            return

        # Calculate new index
        new_idx = current_idx + direction
        if new_idx < 0 or new_idx >= len(channel_names):
            return  # Can't move beyond bounds

        # Swap orders
        other_name = channel_names[new_idx]
        self._channels[name].order, self._channels[other_name].order = \
            self._channels[other_name].order, self._channels[name].order

        # Rebuild UI order
        self._rebuild_channel_list_order()
        self.channels_changed.emit()

    def _rebuild_channel_list_order(self):
        """Rebuild the channel list UI in the correct order."""
        # Get sorted list
        sorted_channels = sorted(self._channels.items(), key=lambda x: x[1].order)

        # Remove all rows from layout (but don't delete them)
        for name in self._channel_rows:
            row = self._channel_rows[name]
            self.channel_list_layout.removeWidget(row)

        # Re-add in sorted order (before the stretch at the end)
        for name, config in sorted_channels:
            if name in self._channel_rows:
                row = self._channel_rows[name]
                self.channel_list_layout.insertWidget(
                    self.channel_list_layout.count() - 1, row
                )

    def _on_channel_settings_requested(self, name: str):
        self.settings_requested.emit(name)
        if name in self._channels and self._channels[name].settings_callback:
            self._channels[name].settings_callback()


# Test the widget standalone
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    # Apply dark theme
    app.setStyleSheet("""
        QWidget {
            background-color: #1e1e1e;
            color: #d4d4d4;
        }
    """)

    # Create test window
    window = QWidget()
    window.setWindowTitle("Channel Manager Test")
    window.setMinimumSize(400, 500)

    layout = QVBoxLayout(window)

    # Create channel manager
    manager = ChannelManagerWidget()
    layout.addWidget(manager)

    # Add test channels - ΔF/F is auto-detected as computed due to name
    manager.set_channels(
        ["Pleth", "ΔF/F", "Thermal Stim", "Isosbestic", "GCaMP", "Laser TTL"]
    )

    # Verify auto-detection worked (use repr to handle unicode)
    print("Auto-detected channel configs:")
    for name, config in manager.get_all_configs().items():
        # Replace delta with 'd' for console output
        safe_name = name.replace('Δ', 'd').replace('δ', 'd')
        print(f"  {safe_name}: type={config.channel_type}, source={config.source}")

    layout.addStretch()

    # Show results on apply
    def on_apply():
        print("Pleth channel:", manager.get_pleth_channel())
        print("Opto Stim channel:", manager.get_opto_stim_channel())
        print("Visible:", manager.get_visible_channels())
        for name, config in manager.get_all_configs().items():
            print(f"  {name}: type={config.channel_type}, visible={config.visible}")

    manager.apply_requested.connect(on_apply)

    window.show()
    sys.exit(app.exec())
