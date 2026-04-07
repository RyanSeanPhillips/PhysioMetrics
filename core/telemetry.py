"""
Anonymous usage tracking and telemetry for PhysioMetrics.

Uses Google Analytics 4 Measurement Protocol for:
- Active user count (session_start / session_end)
- Geographic distribution (from request IP — automatic)
- Key feature usage (file loads, exports, peak detection)

Design: fire-and-forget. If a send fails, we drop it.
No local caching, no retry, no telemetry.log file.
"""

import sys
import platform
import json
import threading
from datetime import datetime

from core.config import (
    get_config_dir,
    get_user_id,
    is_telemetry_enabled
)
from version_info import VERSION_STRING


# ============================================================================
# Configuration
# ============================================================================

GA4_MEASUREMENT_ID = "G-38M0HTXEQ2"
GA4_API_SECRET = "2gmx-luNQFqTNDdZyASmkA"
GA4_SEND_TIMEOUT = 3  # seconds — don't block the UI


# ============================================================================
# Global state
# ============================================================================

_telemetry_initialized = False

_session_data = {
    'files_analyzed': 0,
    'file_types': {'abf': 0, 'smrx': 0, 'edf': 0},
    'total_breaths': 0,
    'total_sweeps': 0,
    'features_used': set(),
    'exports': {},
    'session_start': None,
    'edits_made': 0,
    'edits_added': 0,
    'edits_deleted': 0,
    'last_action': None,
    'timing_data': {},
    'current_file_edits_added': 0,
    'current_file_edits_deleted': 0,
    'current_file_breaths': 0,
}


# ============================================================================
# Heartbeat (for crash tracking — writes to debug_log, NOT telemetry)
# ============================================================================

def _update_heartbeat(action: str, context: str = None):
    """Update heartbeat file and debug log for crash tracking."""
    try:
        from core import error_reporting
        extra_info = {"context": context} if context else None
        error_reporting.update_heartbeat(action, extra_info)
        error_reporting.append_to_debug_log(action, context, level="info")
    except Exception:
        pass


# ============================================================================
# GA4 send — fire and forget
# ============================================================================

def _send_to_ga4(event_name, params=None):
    """
    Send event to GA4 in a background thread. If it fails, drop it.
    No local caching, no retry.
    """
    if not GA4_MEASUREMENT_ID or not GA4_API_SECRET:
        return

    def _do_send():
        try:
            import requests
            url = (f"https://www.google-analytics.com/mp/collect"
                   f"?measurement_id={GA4_MEASUREMENT_ID}"
                   f"&api_secret={GA4_API_SECRET}")
            payload = {
                "client_id": get_user_id(),
                "events": [{"name": event_name, "params": params or {}}]
            }
            requests.post(url, json=payload, timeout=GA4_SEND_TIMEOUT)
        except Exception:
            pass  # Network down, requests missing, timeout — just drop it

    thread = threading.Thread(target=_do_send, daemon=True)
    thread.start()


# ============================================================================
# Initialization
# ============================================================================

def init_telemetry():
    """Initialize telemetry. Call once at app startup."""
    global _telemetry_initialized, _session_data

    if not is_telemetry_enabled():
        return

    try:
        _session_data['session_start'] = datetime.now().isoformat()
        _telemetry_initialized = True

        # Clean up legacy telemetry.log if it exists (older versions cached here)
        try:
            log_file = get_config_dir() / 'telemetry.log'
            if log_file.exists():
                log_file.unlink()
        except Exception:
            pass

        # Send session start — this is the main signal for active user count
        log_event('session_start', {
            'version': VERSION_STRING,
            'platform': sys.platform,
            'python_version': platform.python_version()
        })

        print("Telemetry: Initialized (GA4, fire-and-forget)")

    except Exception as e:
        print(f"Warning: Could not initialize telemetry: {e}")


# ============================================================================
# Public API
# ============================================================================

def log_event(event_name, params=None):
    """
    Log a usage event to GA4. Fire-and-forget.

    GA4 automatically determines geographic location from the request IP.
    No need for explicit geo lookup or X-Forwarded-For headers.
    """
    if not is_telemetry_enabled():
        return

    try:
        if params is None:
            params = {}

        # Add standard params to every event
        params['app_version'] = VERSION_STRING
        params['platform'] = sys.platform
        params['python_version'] = platform.python_version()

        _send_to_ga4(event_name, params)

    except Exception:
        pass


def log_file_loaded(file_type, num_sweeps, num_breaths=None, **extra_params):
    """Log that a file was loaded."""
    global _session_data

    _session_data['files_analyzed'] += 1
    _session_data['file_types'][file_type.lower()] = \
        _session_data['file_types'].get(file_type.lower(), 0) + 1
    _session_data['total_sweeps'] += num_sweeps
    _session_data['last_action'] = f'load_{file_type}'

    if num_breaths:
        _session_data['total_breaths'] += num_breaths

    _session_data['current_file_edits_added'] = 0
    _session_data['current_file_edits_deleted'] = 0
    _session_data['current_file_breaths'] = 0

    _update_heartbeat(f'load_file:{file_type}', f'sweeps={num_sweeps}')

    params = {
        'file_type': file_type,
        'num_sweeps': num_sweeps,
        'num_breaths': num_breaths or 0
    }
    params.update(extra_params)
    log_event('file_loaded', params)


def log_feature_used(feature_name):
    """Log that a feature was used."""
    global _session_data
    _session_data['features_used'].add(feature_name)
    _session_data['last_action'] = feature_name
    _update_heartbeat(f'feature:{feature_name}')
    log_event('feature_used', {'feature': feature_name})


def log_export(export_type):
    """Log that data was exported."""
    global _session_data
    _session_data['exports'][export_type] = \
        _session_data['exports'].get(export_type, 0) + 1
    _session_data['last_action'] = f'export_{export_type}'
    _update_heartbeat(f'export:{export_type}')
    log_event('export', {'export_type': export_type})


def log_file_saved(save_type='npz', eupnea_count=None, sniff_count=None, **extra_params):
    """Log file save with per-file edit metrics for ML evaluation."""
    global _session_data

    num_breaths = _session_data['current_file_breaths']
    edits_added = _session_data['current_file_edits_added']
    edits_deleted = _session_data['current_file_edits_deleted']
    edits_total = edits_added + edits_deleted

    edit_percentage = (edits_total / num_breaths * 100) if num_breaths > 0 else 0
    false_negative_rate = (edits_added / num_breaths * 100) if num_breaths > 0 else 0
    false_positive_rate = (edits_deleted / num_breaths * 100) if num_breaths > 0 else 0

    params = {
        'save_type': save_type,
        'num_breaths': num_breaths,
        'edits_made': edits_total,
        'edits_added': edits_added,
        'edits_deleted': edits_deleted,
        'edit_percentage': round(edit_percentage, 2),
        'false_negative_rate': round(false_negative_rate, 2),
        'false_positive_rate': round(false_positive_rate, 2)
    }

    if eupnea_count is not None and sniff_count is not None:
        params['eupnea_count'] = eupnea_count
        params['sniff_count'] = sniff_count

    params.update(extra_params)
    log_event('file_saved', params)

    print(f"[telemetry] File saved with {edit_percentage:.1f}% edits "
          f"({edits_added} added, {edits_deleted} deleted of {num_breaths} breaths)")


def log_user_engagement():
    """
    Send a lightweight engagement ping to GA4.

    Called every ~45s by a QTimer in MainWindow. Helps GA4 show accurate
    realtime active user count. Fire-and-forget — if it fails, no caching.
    """
    log_event('user_engagement', {})


def log_screen_view(screen_name, screen_class=None, **extra_params):
    """Log when user views a screen/dialog."""
    if not is_telemetry_enabled():
        return
    params = {'screen_name': screen_name}
    if screen_class:
        params['screen_class'] = screen_class
    params.update(extra_params)
    log_event('screen_view', params)


def log_session_end():
    """Log session summary when app closes."""
    if not is_telemetry_enabled():
        return

    try:
        if _session_data['session_start']:
            start = datetime.fromisoformat(_session_data['session_start'])
            duration_minutes = (datetime.now() - start).total_seconds() / 60
        else:
            duration_minutes = 0

        total_breaths = _session_data['total_breaths']
        edits_made = _session_data['edits_made']
        edit_percentage = (edits_made / total_breaths * 100) if total_breaths > 0 else 0

        log_event('session_end', {
            'session_duration_minutes': round(duration_minutes, 1),
            'files_analyzed': _session_data['files_analyzed'],
            'total_breaths': total_breaths,
            'total_sweeps': _session_data['total_sweeps'],
            'features_used_count': len(_session_data['features_used']),
            'exports_count': sum(_session_data['exports'].values()),
            'edits_made': edits_made,
            'edit_percentage': round(edit_percentage, 2)
        })

    except Exception as e:
        print(f"Warning: Could not log session end: {e}")


# ============================================================================
# Detailed tracking (kept for feature usage analytics)
# ============================================================================

def log_timing(operation_name, duration_seconds, **extra_params):
    """Log timing data for operations."""
    global _session_data
    if operation_name not in _session_data['timing_data']:
        _session_data['timing_data'][operation_name] = []
    _session_data['timing_data'][operation_name].append(duration_seconds)

    params = {'operation': operation_name, 'duration_seconds': round(duration_seconds, 2)}
    params.update(extra_params)
    log_event('timing', params)


def log_edit(edit_type, **extra_params):
    """Log manual editing actions."""
    global _session_data
    _session_data['edits_made'] += 1
    _session_data['last_action'] = f'edit_{edit_type}'

    if edit_type == 'add_peak':
        _session_data['edits_added'] += 1
        _session_data['current_file_edits_added'] += 1
    elif edit_type == 'delete_peak':
        _session_data['edits_deleted'] += 1
        _session_data['current_file_edits_deleted'] += 1

    params = {'edit_type': edit_type}
    params.update(extra_params)
    log_event('manual_edit', params)


def log_button_click(button_name, **extra_params):
    """Log button/UI interactions."""
    global _session_data
    _session_data['last_action'] = button_name

    context = ', '.join(f'{k}={v}' for k, v in extra_params.items()) if extra_params else None
    _update_heartbeat(f'button:{button_name}', context)

    params = {'button': button_name}
    params.update(extra_params)
    log_event('button_click', params)


def log_breath_statistics(num_breaths, mean_frequency=None, regularity_score=None, **extra_params):
    """Log breathing analysis statistics."""
    params = {'num_breaths': num_breaths}
    if mean_frequency is not None:
        params['mean_frequency_hz'] = round(mean_frequency, 2)
    if regularity_score is not None:
        params['regularity_score'] = round(regularity_score, 3)
    params.update(extra_params)
    log_event('breath_statistics', params)


def log_crash(error_message, **extra_params):
    """Log application crash/error to GA4."""
    global _session_data
    params = {
        'error_type': error_message,
        'last_action': _session_data.get('last_action', 'unknown')
    }
    params.update(extra_params)
    log_event('crash', params)


def log_warning(warning_message, **extra_params):
    """Log non-critical warnings."""
    params = {'warning_type': warning_message}
    params.update(extra_params)
    log_event('warning', params)


def log_filter_applied(filter_type, **params):
    """Log filter application with settings."""
    telemetry_params = {'filter_type': filter_type}
    telemetry_params.update(params)
    log_event('filter_applied', telemetry_params)


def log_peak_detection(method, num_peaks, **params):
    """Log peak detection results."""
    global _session_data
    num_breaths = params.get('num_breaths', num_peaks)
    _session_data['current_file_breaths'] = num_breaths

    telemetry_params = {'detection_method': method, 'num_peaks': num_peaks}
    telemetry_params.update(params)
    log_event('peak_detection', telemetry_params)


def log_navigation(action, **params):
    """Log navigation actions."""
    telemetry_params = {'navigation_action': action}
    telemetry_params.update(params)
    log_event('navigation', telemetry_params)


def log_keyboard_shortcut(shortcut_name, **params):
    """Log keyboard shortcut usage."""
    telemetry_params = {'shortcut': shortcut_name}
    telemetry_params.update(params)
    log_event('keyboard_shortcut', telemetry_params)


def log_error(error, context=None):
    """Log an error or exception to GA4."""
    if not is_telemetry_enabled():
        return
    try:
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error)[:100],
        }
        if context:
            error_data.update(context)
        log_event('error', error_data)
    except Exception:
        pass


# ============================================================================
# Session data access (for crash reporting)
# ============================================================================

def get_session_data() -> dict:
    """Get a copy of current session data for crash reports."""
    return {
        'files_analyzed': _session_data.get('files_analyzed', 0),
        'total_breaths': _session_data.get('total_breaths', 0),
        'total_sweeps': _session_data.get('total_sweeps', 0),
        'features_used': list(_session_data.get('features_used', set())),
        'last_action': _session_data.get('last_action', 'unknown'),
        'session_start': _session_data.get('session_start'),
        'edits_made': _session_data.get('edits_made', 0),
        'edits_added': _session_data.get('edits_added', 0),
        'edits_deleted': _session_data.get('edits_deleted', 0),
    }


def is_active():
    """Check if telemetry is initialized and enabled."""
    return _telemetry_initialized and is_telemetry_enabled()
