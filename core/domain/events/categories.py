"""
Event category definitions and registry.

This module defines event categories (types) and provides a registry
for managing built-in and custom categories.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
import copy

from .models import MarkerType


@dataclass
class EventCategory:
    """
    Definition of an event category (type).

    Categories group related event types together and define their
    default appearance and behavior.

    Attributes:
        name: Internal key ('respiratory', 'behavior', etc.)
        display_name: Human-readable name for UI
        color: Default hex color for markers of this category
        line_style: Line style ('solid', 'dashed', 'dotted')
        line_width: Line width in pixels
        default_marker_type: Default marker type when adding
        default_detection_method: Default detection method
        channel_hints: Keywords to auto-detect relevant channels
        default_detection_params: Default parameters for auto-detection
        is_builtin: Whether this is a built-in category
        usage_count: For sorting by frequency of use
        labels: Available labels within this category
    """

    # Identity
    name: str
    display_name: str

    # Appearance
    color: str = "#00CED1"
    line_style: str = "solid"
    line_width: float = 1.5

    # Behavior
    default_marker_type: MarkerType = MarkerType.SINGLE
    default_detection_method: str = "manual"
    channel_hints: List[str] = field(default_factory=list)
    default_detection_params: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    is_builtin: bool = False
    usage_count: int = 0

    # Labels within this category
    labels: List[str] = field(default_factory=list)

    def get_display_label(self, label: str) -> str:
        """Get human-readable version of a label."""
        return label.replace('_', ' ').title()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'display_name': self.display_name,
            'color': self.color,
            'line_style': self.line_style,
            'line_width': self.line_width,
            'default_marker_type': self.default_marker_type.value,
            'default_detection_method': self.default_detection_method,
            'channel_hints': self.channel_hints,
            'default_detection_params': self.default_detection_params,
            'is_builtin': self.is_builtin,
            'usage_count': self.usage_count,
            'labels': self.labels,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventCategory':
        """Create EventCategory from dictionary."""
        marker_type = data.get('default_marker_type', 'single')
        if isinstance(marker_type, str):
            marker_type = MarkerType(marker_type)

        return cls(
            name=data['name'],
            display_name=data.get('display_name', data['name'].title()),
            color=data.get('color', '#00CED1'),
            line_style=data.get('line_style', 'solid'),
            line_width=data.get('line_width', 1.5),
            default_marker_type=marker_type,
            default_detection_method=data.get('default_detection_method', 'manual'),
            channel_hints=data.get('channel_hints', []),
            default_detection_params=data.get('default_detection_params', {}),
            is_builtin=data.get('is_builtin', False),
            usage_count=data.get('usage_count', 0),
            labels=data.get('labels', []),
        )

    def copy(self) -> 'EventCategory':
        """Create a deep copy of this category."""
        return EventCategory.from_dict(self.to_dict())


# Built-in category definitions
# Simplified to core categories: Hargreaves, Stimulus, Movement, Artifact, Custom
BUILTIN_CATEGORIES: Dict[str, EventCategory] = {
    'hargreaves': EventCategory(
        name='hargreaves',
        display_name='Hargreaves',
        color='#F44336',  # Red
        default_marker_type=MarkerType.PAIRED,
        labels=['heat_onset', 'withdrawal', 'test'],
        channel_hints=['temp', 'thermal', 'hargreaves'],
        is_builtin=True,
    ),
    'stimulus': EventCategory(
        name='stimulus',
        display_name='Stimulus',
        color='#FF9800',  # Orange
        default_marker_type=MarkerType.SINGLE,
        labels=['stim_on', 'stim_off', 'ttl'],
        channel_hints=['stim', 'ttl', 'trigger'],
        default_detection_method='ttl',
        is_builtin=True,
    ),
    'movement': EventCategory(
        name='movement',
        display_name='Movement',
        color='#2196F3',  # Blue
        default_marker_type=MarkerType.PAIRED,
        labels=['movement', 'grooming', 'sniffing'],
        channel_hints=['motion', 'movement', 'accel'],
        is_builtin=True,
    ),
    'artifact': EventCategory(
        name='artifact',
        display_name='Artifact',
        color='#9E9E9E',  # Gray
        default_marker_type=MarkerType.PAIRED,
        labels=['artifact', 'noise', 'exclude'],
        is_builtin=True,
    ),
    'custom': EventCategory(
        name='custom',
        display_name='Custom',
        color='#9C27B0',  # Purple
        default_marker_type=MarkerType.SINGLE,
        labels=['marker', 'event'],
        is_builtin=True,
    ),
}


class CategoryRegistry:
    """
    Registry for managing event categories.

    Handles both built-in and custom categories, supports loading/saving
    custom categories, and provides lookup methods.
    """

    def __init__(self):
        # Start with copies of built-in categories
        self._categories: Dict[str, EventCategory] = {}
        self._reset_to_defaults()

    def _reset_to_defaults(self) -> None:
        """Reset to built-in categories only."""
        self._categories = {
            name: cat.copy() for name, cat in BUILTIN_CATEGORIES.items()
        }

    def get(self, name: str) -> Optional[EventCategory]:
        """Get a category by name."""
        return self._categories.get(name)

    def get_or_default(self, name: str) -> EventCategory:
        """Get a category by name, falling back to 'custom' if not found."""
        return self._categories.get(name, self._categories['custom'])

    def add(self, category: EventCategory) -> None:
        """Add or update a category."""
        self._categories[category.name] = category

    def remove(self, name: str) -> bool:
        """
        Remove a category by name.

        Built-in categories cannot be removed.

        Returns:
            True if removed, False if built-in or not found
        """
        cat = self._categories.get(name)
        if cat and not cat.is_builtin:
            del self._categories[name]
            return True
        return False

    def get_all(self) -> List[EventCategory]:
        """Get all categories, sorted by name."""
        return sorted(self._categories.values(), key=lambda c: c.name)

    def get_builtin(self) -> List[EventCategory]:
        """Get only built-in categories."""
        return [c for c in self._categories.values() if c.is_builtin]

    def get_custom(self) -> List[EventCategory]:
        """Get only custom (user-created) categories."""
        return [c for c in self._categories.values() if not c.is_builtin]

    def get_by_usage(self) -> List[EventCategory]:
        """Get all categories sorted by usage count (descending)."""
        return sorted(self._categories.values(), key=lambda c: -c.usage_count)

    def increment_usage(self, name: str) -> None:
        """Increment usage count for a category."""
        if name in self._categories:
            self._categories[name].usage_count += 1

    def add_label(self, category_name: str, label: str) -> bool:
        """
        Add a label to a category.

        Returns:
            True if added, False if category not found or label exists
        """
        cat = self._categories.get(category_name)
        if cat and label not in cat.labels:
            cat.labels.append(label)
            return True
        return False

    def remove_label(self, category_name: str, label: str) -> bool:
        """
        Remove a label from a category.

        Returns:
            True if removed, False if not found
        """
        cat = self._categories.get(category_name)
        if cat and label in cat.labels:
            cat.labels.remove(label)
            return True
        return False

    def find_category_for_channel(self, channel_name: str) -> Optional[EventCategory]:
        """
        Find a category that matches channel hints.

        Args:
            channel_name: Name of the channel

        Returns:
            Matching category or None
        """
        channel_lower = channel_name.lower()
        for cat in self._categories.values():
            for hint in cat.channel_hints:
                if hint.lower() in channel_lower:
                    return cat
        return None

    def get_all_labels(self) -> Dict[str, List[str]]:
        """
        Get all labels grouped by category.

        Returns:
            Dict mapping category name to list of labels
        """
        return {
            cat.name: cat.labels
            for cat in self._categories.values()
        }

    def get_flat_label_list(self) -> List[tuple]:
        """
        Get flat list of (category_name, label) tuples.

        Returns:
            List of (category_name, label) tuples sorted by category
        """
        result = []
        for cat in sorted(self._categories.values(), key=lambda c: c.name):
            for label in cat.labels:
                result.append((cat.name, label))
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary for serialization."""
        return {
            'categories': {
                name: cat.to_dict()
                for name, cat in self._categories.items()
            }
        }

    def to_json(self) -> str:
        """Convert registry to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CategoryRegistry':
        """Create registry from dictionary."""
        registry = cls()
        if 'categories' in data:
            for name, cat_data in data['categories'].items():
                registry._categories[name] = EventCategory.from_dict(cat_data)
        return registry

    @classmethod
    def from_json(cls, json_str: str) -> 'CategoryRegistry':
        """Create registry from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_custom_only_dict(self) -> Dict[str, Any]:
        """
        Get only custom categories for saving.

        Built-in categories don't need to be saved since they're
        always loaded from code.
        """
        return {
            'custom_categories': [
                cat.to_dict()
                for cat in self._categories.values()
                if not cat.is_builtin
            ],
            'builtin_customizations': {
                name: {
                    'labels': cat.labels,
                    'color': cat.color,
                    'usage_count': cat.usage_count,
                }
                for name, cat in self._categories.items()
                if cat.is_builtin and (
                    cat.labels != BUILTIN_CATEGORIES[name].labels or
                    cat.color != BUILTIN_CATEGORIES[name].color or
                    cat.usage_count > 0
                )
            }
        }

    def load_customizations(self, data: Dict[str, Any]) -> None:
        """
        Load custom categories and customizations.

        Args:
            data: Dict with 'custom_categories' and 'builtin_customizations'
        """
        # Load custom categories
        for cat_data in data.get('custom_categories', []):
            self._categories[cat_data['name']] = EventCategory.from_dict(cat_data)

        # Apply customizations to built-in categories
        for name, customization in data.get('builtin_customizations', {}).items():
            if name in self._categories:
                cat = self._categories[name]
                if 'labels' in customization:
                    cat.labels = customization['labels']
                if 'color' in customization:
                    cat.color = customization['color']
                if 'usage_count' in customization:
                    cat.usage_count = customization['usage_count']


# Global singleton registry
_registry: Optional[CategoryRegistry] = None


def get_category_registry() -> CategoryRegistry:
    """Get the global category registry singleton."""
    global _registry
    if _registry is None:
        _registry = CategoryRegistry()
    return _registry


def reset_category_registry() -> CategoryRegistry:
    """Reset the global registry to defaults (for testing)."""
    global _registry
    _registry = CategoryRegistry()
    return _registry
