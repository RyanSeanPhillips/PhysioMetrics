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


# ============================================================================
# Crash Reports and Error Logging Directories
# ============================================================================

def get_crash_reports_dir():
    """
    Get crash reports directory.

    Returns:
        Path: Crash reports directory (created if needed)
    """
    crash_dir = get_config_dir() / 'crash_reports'
    crash_dir.mkdir(exist_ok=True)
    return crash_dir


def get_error_log_path():
    """
    Get path to error log file.

    Returns:
        Path: Error log file path
    """
    return get_config_dir() / 'error_log.json'


# ============================================================================
# NPZ File Registry
# ============================================================================

def get_npz_registry_path():
    """
    Get path to NPZ registry file.

    The registry maps source data files (e.g., FP_data*.csv) to their
    processed NPZ file paths, enabling automatic discovery when the user
    opens a source file that was previously processed.

    Returns:
        Path: NPZ registry file path
    """
    return get_config_dir() / 'npz_registry.json'


def load_npz_registry():
    """
    Load the NPZ file registry.

    Returns:
        dict: Registry mapping source_file_path -> {
            'npz_path': str,
            'created': str (ISO timestamp),
            'last_accessed': str (ISO timestamp),
            'n_experiments': int,
            'animal_ids': list[str]
        }
    """
    registry_path = get_npz_registry_path()
    if registry_path.exists():
        try:
            with open(registry_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_npz_registry(registry):
    """
    Save the NPZ file registry.

    Args:
        registry: dict mapping source paths to NPZ info
    """
    registry_path = get_npz_registry_path()
    try:
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
    except IOError as e:
        print(f"[Config] Warning: Failed to save NPZ registry: {e}")


def register_npz_file(source_file_path, npz_path, n_experiments=1, animal_ids=None):
    """
    Register an NPZ file in the registry, mapping it to its source data file.

    Args:
        source_file_path: Path to the original source file (e.g., FP_data*.csv)
        npz_path: Path to the saved NPZ file
        n_experiments: Number of experiments in the NPZ
        animal_ids: Optional list of animal IDs for each experiment
    """
    from datetime import datetime

    registry = load_npz_registry()

    # Normalize paths to strings
    source_key = str(Path(source_file_path).resolve())
    npz_str = str(Path(npz_path).resolve())

    now = datetime.now().isoformat()

    registry[source_key] = {
        'npz_path': npz_str,
        'created': registry.get(source_key, {}).get('created', now),
        'last_accessed': now,
        'n_experiments': n_experiments,
        'animal_ids': animal_ids or []
    }

    save_npz_registry(registry)
    print(f"[Config] Registered NPZ: {Path(source_file_path).name} -> {Path(npz_path).name}")


def lookup_npz_for_source(source_file_path):
    """
    Look up the NPZ file path for a given source data file.

    Args:
        source_file_path: Path to the source file to look up

    Returns:
        dict with 'npz_path' and metadata if found, None if not registered
        or if the NPZ file no longer exists
    """
    registry = load_npz_registry()

    source_key = str(Path(source_file_path).resolve())

    if source_key in registry:
        entry = registry[source_key]
        npz_path = Path(entry['npz_path'])

        # Verify the NPZ file still exists
        if npz_path.exists():
            # Update last accessed time
            from datetime import datetime
            entry['last_accessed'] = datetime.now().isoformat()
            save_npz_registry(registry)
            return entry
        else:
            # NPZ file was deleted, remove from registry
            print(f"[Config] NPZ file no longer exists, removing from registry: {npz_path}")
            del registry[source_key]
            save_npz_registry(registry)

    return None


def get_recent_npz_files(limit=10):
    """
    Get list of recently accessed NPZ files.

    Args:
        limit: Maximum number of entries to return

    Returns:
        List of dicts with 'source_path', 'npz_path', 'last_accessed', etc.
        sorted by last_accessed (most recent first)
    """
    registry = load_npz_registry()

    # Filter to only existing files and add source_path key
    valid_entries = []
    for source_path, entry in registry.items():
        npz_path = Path(entry['npz_path'])
        if npz_path.exists():
            valid_entries.append({
                'source_path': source_path,
                **entry
            })

    # Sort by last_accessed (most recent first)
    valid_entries.sort(key=lambda x: x.get('last_accessed', ''), reverse=True)

    return valid_entries[:limit]


def cleanup_npz_registry():
    """
    Remove entries for NPZ files that no longer exist.

    Returns:
        int: Number of entries removed
    """
    registry = load_npz_registry()
    original_count = len(registry)

    # Keep only entries where NPZ file exists
    cleaned = {
        source: entry
        for source, entry in registry.items()
        if Path(entry['npz_path']).exists()
    }

    removed = original_count - len(cleaned)
    if removed > 0:
        save_npz_registry(cleaned)
        print(f"[Config] Cleaned up NPZ registry: removed {removed} stale entries")

    return removed
