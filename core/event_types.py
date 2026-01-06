"""
Event type registry and presets for PhysioMetrics event markers.

This module manages event marker types, their colors, detection methods,
and persistence across sessions.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

from core.config import get_config_dir


# ============================================================================
# Event Type Data Structures
# ============================================================================

@dataclass
class EventTypeConfig:
    """Configuration for an event marker type."""

    name: str                           # Internal key (e.g., 'lick_bout')
    display_name: str                   # User-visible name (e.g., 'Lick Bout')
    color: str                          # Hex color (e.g., '#00CED1')
    detection_method: str               # 'manual', 'threshold', 'hargreaves_thermal', etc.
    channel_hints: List[str] = field(default_factory=list)  # Auto-select channel hints
    default_params: Dict[str, Any] = field(default_factory=dict)  # Detection parameters
    usage_count: int = 0                # For sorting by frequency
    is_builtin: bool = False            # Built-in types cannot be deleted

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'EventTypeConfig':
        """Create from dictionary."""
        return cls(**data)


# ============================================================================
# Built-in Event Type Presets
# ============================================================================

BUILTIN_EVENT_TYPES: Dict[str, EventTypeConfig] = {
    'lick_bout': EventTypeConfig(
        name='lick_bout',
        display_name='Lick Bout',
        color='#00CED1',  # Dark cyan
        detection_method='threshold',
        channel_hints=['lick', 'bout', 'detector'],
        default_params={
            'threshold': 0.5,
            'min_duration_ms': 50,
            'min_gap_s': 1.0
        },
        is_builtin=True
    ),
    'hargreaves': EventTypeConfig(
        name='hargreaves',
        display_name='Hargreaves',
        color='#FF6B6B',  # Coral red
        detection_method='hargreaves_thermal',
        channel_hints=['heat', 'thermal', 'hargreaves', 'temp'],
        default_params={
            'noise_sigma': 3,
            'search_window_s': 15
        },
        is_builtin=True
    ),
    'stim': EventTypeConfig(
        name='stim',
        display_name='Stimulus',
        color='#FFD700',  # Gold
        detection_method='ttl_rising',
        channel_hints=['stim', 'ttl', 'trigger', 'sync'],
        default_params={
            'threshold': 2.5,
            'edge': 'rising'
        },
        is_builtin=True
    ),
    'sigh': EventTypeConfig(
        name='sigh',
        display_name='Sigh',
        color='#9370DB',  # Medium purple
        detection_method='manual',
        channel_hints=[],
        default_params={},
        is_builtin=True
    ),
    'sniff': EventTypeConfig(
        name='sniff',
        display_name='Sniff',
        color='#32CD32',  # Lime green
        detection_method='manual',
        channel_hints=[],
        default_params={},
        is_builtin=True
    ),
    'apnea': EventTypeConfig(
        name='apnea',
        display_name='Apnea',
        color='#FF4444',  # Red
        detection_method='manual',
        channel_hints=[],
        default_params={},
        is_builtin=True
    ),
    'artifact': EventTypeConfig(
        name='artifact',
        display_name='Artifact',
        color='#808080',  # Gray
        detection_method='manual',
        channel_hints=[],
        default_params={},
        is_builtin=True
    ),
}


# ============================================================================
# Event Type Registry
# ============================================================================

class EventTypeRegistry:
    """
    Manages event marker types with persistence.

    Types are stored globally in app settings and also saved per-project.
    Types are sorted by usage frequency for quick access.
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern - one registry for the app."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._types: Dict[str, EventTypeConfig] = {}
        self._initialized = True
        self._load_types()

    def _get_config_file(self) -> Path:
        """Get path to event types config file."""
        return get_config_dir() / 'event_types.json'

    def _load_types(self):
        """Load event types from config file, merging with built-ins."""
        # Start with built-in types
        self._types = {k: EventTypeConfig(**asdict(v)) for k, v in BUILTIN_EVENT_TYPES.items()}

        # Load custom types from config
        config_file = self._get_config_file()
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    saved_data = json.load(f)

                # Load custom types
                for name, type_data in saved_data.get('custom_types', {}).items():
                    if name not in BUILTIN_EVENT_TYPES:
                        self._types[name] = EventTypeConfig.from_dict(type_data)

                # Update usage counts for built-in types
                for name, count in saved_data.get('usage_counts', {}).items():
                    if name in self._types:
                        self._types[name].usage_count = count

            except Exception as e:
                print(f"Warning: Could not load event types config: {e}")

    def _save_types(self):
        """Save custom types and usage counts to config file."""
        try:
            # Separate custom types from built-ins
            custom_types = {
                name: config.to_dict()
                for name, config in self._types.items()
                if not config.is_builtin
            }

            # Save usage counts for all types
            usage_counts = {
                name: config.usage_count
                for name, config in self._types.items()
            }

            config_file = self._get_config_file()
            with open(config_file, 'w') as f:
                json.dump({
                    'custom_types': custom_types,
                    'usage_counts': usage_counts
                }, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save event types config: {e}")

    def get_all_types(self) -> List[EventTypeConfig]:
        """Get all event types, sorted by usage frequency."""
        return sorted(
            self._types.values(),
            key=lambda t: (-t.usage_count, t.display_name)
        )

    def get_type(self, name: str) -> Optional[EventTypeConfig]:
        """Get a specific event type by name."""
        return self._types.get(name)

    def get_type_names(self) -> List[str]:
        """Get list of type names, sorted by usage frequency."""
        return [t.name for t in self.get_all_types()]

    def add_type(self, config: EventTypeConfig) -> bool:
        """
        Add a new custom event type.

        Args:
            config: EventTypeConfig for the new type

        Returns:
            True if added, False if name already exists
        """
        if config.name in self._types:
            return False

        config.is_builtin = False  # Ensure custom types are deletable
        self._types[config.name] = config
        self._save_types()
        return True

    def update_type(self, name: str, **kwargs) -> bool:
        """
        Update an existing event type.

        Args:
            name: Type name to update
            **kwargs: Fields to update (display_name, color, etc.)

        Returns:
            True if updated, False if not found
        """
        if name not in self._types:
            return False

        config = self._types[name]
        for key, value in kwargs.items():
            if hasattr(config, key) and key != 'name' and key != 'is_builtin':
                setattr(config, key, value)

        self._save_types()
        return True

    def delete_type(self, name: str) -> bool:
        """
        Delete a custom event type.

        Args:
            name: Type name to delete

        Returns:
            True if deleted, False if not found or is built-in
        """
        if name not in self._types:
            return False

        if self._types[name].is_builtin:
            return False  # Cannot delete built-in types

        del self._types[name]
        self._save_types()
        return True

    def increment_usage(self, name: str):
        """Increment usage count for a type (for frequency sorting)."""
        if name in self._types:
            self._types[name].usage_count += 1
            self._save_types()

    def get_color(self, name: str) -> str:
        """Get color for an event type, with fallback."""
        config = self._types.get(name)
        return config.color if config else '#808080'  # Gray fallback

    def get_display_name(self, name: str) -> str:
        """Get display name for an event type."""
        config = self._types.get(name)
        return config.display_name if config else name

    def find_type_for_channel(self, channel_name: str) -> Optional[str]:
        """
        Find a suitable event type based on channel name hints.

        Args:
            channel_name: Name of the channel

        Returns:
            Type name if a match is found, None otherwise
        """
        channel_lower = channel_name.lower()

        for type_config in self.get_all_types():
            for hint in type_config.channel_hints:
                if hint.lower() in channel_lower:
                    return type_config.name

        return None

    def export_for_project(self) -> Dict:
        """
        Export type definitions for saving in project file.

        Returns:
            Dict of type configurations used in project
        """
        return {
            name: config.to_dict()
            for name, config in self._types.items()
        }

    def import_from_project(self, type_data: Dict):
        """
        Import type definitions from a project file.

        This adds any custom types from the project that don't exist locally.

        Args:
            type_data: Dict of type configurations from project
        """
        for name, config_dict in type_data.items():
            if name not in self._types and name not in BUILTIN_EVENT_TYPES:
                try:
                    config = EventTypeConfig.from_dict(config_dict)
                    config.is_builtin = False
                    self._types[name] = config
                except Exception as e:
                    print(f"Warning: Could not import event type '{name}': {e}")

        self._save_types()


# ============================================================================
# Module-level convenience functions
# ============================================================================

def get_registry() -> EventTypeRegistry:
    """Get the singleton event type registry."""
    return EventTypeRegistry()


def get_event_color(event_type: str) -> str:
    """Get color for an event type."""
    return get_registry().get_color(event_type)


def get_event_display_name(event_type: str) -> str:
    """Get display name for an event type."""
    return get_registry().get_display_name(event_type)


def get_all_event_types() -> List[EventTypeConfig]:
    """Get all event types sorted by usage frequency."""
    return get_registry().get_all_types()


def create_custom_type(
    name: str,
    display_name: str,
    color: str,
    detection_method: str = 'manual',
    channel_hints: List[str] = None,
    default_params: Dict = None
) -> bool:
    """
    Create a new custom event type.

    Args:
        name: Internal name (lowercase, no spaces)
        display_name: User-visible name
        color: Hex color string
        detection_method: Detection method name
        channel_hints: List of channel name hints
        default_params: Default detection parameters

    Returns:
        True if created, False if name exists
    """
    config = EventTypeConfig(
        name=name,
        display_name=display_name,
        color=color,
        detection_method=detection_method,
        channel_hints=channel_hints or [],
        default_params=default_params or {},
        is_builtin=False
    )
    return get_registry().add_type(config)
