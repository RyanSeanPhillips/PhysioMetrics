"""
Persistent configuration management for PhysioMetrics.

Handles user preferences, UUID generation, and config file storage.
"""

import os
import json
import uuid
from pathlib import Path


def get_config_dir():
    """
    Get platform-specific config directory.

    Returns:
        Path: Config directory path

    Platform paths:
    - Windows: C:/Users/{username}/AppData/Roaming/PhysioMetrics
    - Mac: ~/Library/Application Support/PhysioMetrics
    - Linux: ~/.config/PhysioMetrics
    """
    import sys

    if sys.platform == 'win32':
        # Windows: AppData/Roaming
        base = os.environ.get('APPDATA', os.path.expanduser('~'))
        config_dir = Path(base) / 'PhysioMetrics'
    elif sys.platform == 'darwin':
        # macOS: ~/Library/Application Support
        config_dir = Path.home() / 'Library' / 'Application Support' / 'PhysioMetrics'
    else:
        # Linux: ~/.config
        config_dir = Path.home() / '.config' / 'PhysioMetrics'

    # Create directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)

    return config_dir


def get_config_path():
    """
    Get full path to config file.

    Returns:
        Path: Config file path (e.g., ~/.config/PhysioMetrics/config.json)
    """
    return get_config_dir() / 'config.json'


def generate_user_id():
    """
    Generate a random UUID for anonymous user identification.

    Returns:
        str: Random UUID (e.g., 'a3f2e8c9-4b7d-...')
    """
    return str(uuid.uuid4())


def load_config():
    """
    Load config from file. Returns default config if file doesn't exist.

    Returns:
        dict: Config dictionary with keys:
            - user_id (str): Anonymous UUID
            - telemetry_enabled (bool): Whether to send usage data
            - crash_reports_enabled (bool): Whether to send crash reports
            - first_launch (bool): Whether this is first launch
    """
    config_path = get_config_path()

    # Default config for first launch
    default_config = {
        'user_id': generate_user_id(),
        'telemetry_enabled': True,  # Opt-out model
        'crash_reports_enabled': True,
        'first_launch': True,
        'version': '1.0.9'
    }

    # Try to load existing config
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Merge with defaults (in case new keys added in update)
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value

            # Mark not first launch
            config['first_launch'] = False

            return config

        except Exception as e:
            print(f"Warning: Could not load config: {e}")
            # Return default config on error
            return default_config
    else:
        # First launch - return default config
        return default_config


def save_config(config):
    """
    Save config to file.

    Args:
        config (dict): Config dictionary to save

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        config_path = get_config_path()

        with open(config_path, 'w') as f:
            json.dump(config, indent=2, fp=f)

        return True

    except Exception as e:
        print(f"Warning: Could not save config: {e}")
        return False


def update_config(key, value):
    """
    Update a single config value and save.

    Args:
        key (str): Config key to update
        value: New value

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        config = load_config()
        config[key] = value
        return save_config(config)
    except Exception:
        return False


# Convenience functions
def is_first_launch():
    """Check if this is the first launch on this computer."""
    return load_config().get('first_launch', True)


def get_user_id():
    """Get the anonymous user ID."""
    return load_config().get('user_id', generate_user_id())


def is_telemetry_enabled():
    """Check if telemetry is enabled."""
    return load_config().get('telemetry_enabled', True)


def is_crash_reports_enabled():
    """Check if crash reports are enabled."""
    return load_config().get('crash_reports_enabled', True)


def set_telemetry_enabled(enabled):
    """Enable or disable telemetry."""
    return update_config('telemetry_enabled', enabled)


def set_crash_reports_enabled(enabled):
    """Enable or disable crash reports."""
    return update_config('crash_reports_enabled', enabled)


def mark_first_launch_complete():
    """Mark that first launch has been completed."""
    return update_config('first_launch', False)
