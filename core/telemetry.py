"""
Anonymous usage tracking and telemetry for PlethApp.

Uses dual system:
- Google Analytics 4: Usage statistics (unlimited events)
- Sentry: Crash reports with stack traces (limited to 5k/month)

No personal information, file names, or experimental data is collected.
"""

import sys
import platform
import json
import threading
from datetime import datetime
from pathlib import Path

from core.config import (
    get_config_dir,
    get_user_id,
    is_telemetry_enabled,
    is_crash_reports_enabled
)
from version_info import VERSION_STRING


# ============================================================================
# CONFIGURATION - Add your credentials here
# ============================================================================

# Google Analytics 4 Measurement Protocol
# Get these from: https://analytics.google.com/
# Admin → Data Streams → Your stream → Measurement Protocol API secrets
GA4_MEASUREMENT_ID = "G-38M0HTXEQ2"
GA4_API_SECRET = "2gmx-luNQFqTNDdZyASmkA"

# Sentry DSN (for crash reports)
# Get from: https://sentry.io/settings/projects/your-project/keys/
SENTRY_DSN = "https://3a2829e5a500579ba0f205028b68645c@o4510270639898624.ingest.us.sentry.io/4510270680596480"


# ============================================================================
# Global telemetry state
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
}


# ============================================================================
# Initialization
# ============================================================================

def init_telemetry():
    """
    Initialize telemetry system.

    Call this once at app startup (after first-launch dialog).
    """
    global _telemetry_initialized, _session_data

    if not is_telemetry_enabled():
        return

    try:
        # Initialize Sentry (for crash reports)
        if is_crash_reports_enabled():
            _init_sentry()

        # Reset session data
        _session_data['session_start'] = datetime.now().isoformat()
        _telemetry_initialized = True

        # Send session start event to GA4
        log_event('session_start', {
            'version': VERSION_STRING,
            'platform': sys.platform,
            'python_version': platform.python_version()
        })

        print("Telemetry: Initialized (GA4 + Sentry)")

    except Exception as e:
        print(f"Warning: Could not initialize telemetry: {e}")


def _init_sentry():
    """Initialize Sentry SDK for crash reports (optional)."""
    try:
        import sentry_sdk

        if SENTRY_DSN:
            sentry_sdk.init(
                dsn=SENTRY_DSN,
                traces_sample_rate=0.0,  # No performance tracing (saves quota)
                send_default_pii=False,  # Never send personal info
                before_send=_sanitize_sentry_event,
            )

            # Set anonymous user ID
            sentry_sdk.set_user({"id": get_user_id()})

            # Set app context
            sentry_sdk.set_context("app", {
                "version": VERSION_STRING,
                "platform": sys.platform,
                "python_version": platform.python_version(),
            })

            print("Telemetry: Sentry initialized for crash reports")

    except ImportError:
        print("Telemetry: Sentry not installed (crash reports disabled)")
    except Exception as e:
        print(f"Warning: Sentry initialization failed: {e}")


def _sanitize_sentry_event(event, hint):
    """
    Sanitize event before sending to Sentry.

    Remove any file paths, personal info, or experimental data.
    """
    # Remove file paths from stack traces
    if 'exception' in event:
        for exc in event['exception'].get('values', []):
            if 'stacktrace' in exc:
                for frame in exc['stacktrace'].get('frames', []):
                    # Keep only filename, not full path
                    if 'abs_path' in frame:
                        frame['abs_path'] = Path(frame['abs_path']).name
                    if 'filename' in frame:
                        frame['filename'] = Path(frame['filename']).name

    return event


# ============================================================================
# Google Analytics 4 Integration
# ============================================================================

def _send_to_google_analytics(event_name, params=None):
    """
    Send event to Google Analytics 4 in background thread (non-blocking).

    Args:
        event_name (str): Event name (e.g., "file_loaded", "gmm_clustering")
        params (dict, optional): Event parameters (must not contain PII)

    Docs: https://developers.google.com/analytics/devguides/collection/protocol/ga4

    Note: This function returns immediately. Network call happens in background.
    """
    if not GA4_MEASUREMENT_ID or not GA4_API_SECRET:
        # GA4 not configured - log locally
        _log_event_locally({
            'event': event_name,
            'params': params,
            'timestamp': datetime.now().isoformat()
        })
        return

    # Send in background thread (non-blocking)
    thread = threading.Thread(
        target=_send_to_ga4_blocking,
        args=(event_name, params),
        daemon=True  # Don't block app exit
    )
    thread.start()  # Returns immediately


def _send_to_ga4_blocking(event_name, params):
    """
    Actually send event to GA4 (runs in background thread).

    This function blocks on network I/O, but it runs in a background thread
    so it doesn't affect UI responsiveness.
    """
    try:
        import requests

        url = f"https://www.google-analytics.com/mp/collect?measurement_id={GA4_MEASUREMENT_ID}&api_secret={GA4_API_SECRET}"

        # Build payload
        payload = {
            "client_id": get_user_id(),  # Anonymous UUID
            "events": [{
                "name": event_name,
                "params": params or {}
            }]
        }

        # Send to GA4 (blocking, but in background thread)
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()

    except ImportError:
        # requests not available - log locally
        _log_event_locally({
            'event': event_name,
            'params': params,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        # Network error or timeout - log locally (never interrupt user)
        print(f"Telemetry: GA4 send failed (logged locally): {e}")
        _log_event_locally({
            'event': event_name,
            'params': params,
            'timestamp': datetime.now().isoformat()
        })


def _log_event_locally(event):
    """
    Log event to local file (fallback when GA4 unavailable).

    Args:
        event (dict): Event data
    """
    try:
        log_file = get_config_dir() / 'telemetry.log'

        with open(log_file, 'a') as f:
            f.write(json.dumps(event) + '\n')

    except Exception:
        # Silently fail - never interrupt user workflow
        pass


# ============================================================================
# Public API - Usage Statistics (sent to Google Analytics)
# ============================================================================

def log_event(event_name, params=None):
    """
    Log a usage event to Google Analytics 4.

    Args:
        event_name (str): Event name (e.g., "file_loaded", "gmm_clustering")
        params (dict, optional): Event parameters (must not contain PII)

    Example:
        log_event("file_loaded", {
            "file_type": "abf",
            "num_sweeps": 10,
            "num_breaths": 247
        })
    """
    if not is_telemetry_enabled():
        return

    try:
        # Add standard parameters
        if params is None:
            params = {}

        params['version'] = VERSION_STRING
        params['platform'] = sys.platform

        # Send to Google Analytics
        _send_to_google_analytics(event_name, params)

    except Exception as e:
        # Never crash app due to telemetry
        print(f"Warning: Telemetry event failed: {e}")


def log_file_loaded(file_type, num_sweeps, num_breaths=None):
    """
    Log that a file was loaded.

    Args:
        file_type (str): File extension ('abf', 'smrx', 'edf')
        num_sweeps (int): Number of sweeps in file
        num_breaths (int, optional): Number of breaths detected
    """
    global _session_data

    _session_data['files_analyzed'] += 1
    _session_data['file_types'][file_type.lower()] = \
        _session_data['file_types'].get(file_type.lower(), 0) + 1
    _session_data['total_sweeps'] += num_sweeps

    if num_breaths:
        _session_data['total_breaths'] += num_breaths

    log_event('file_loaded', {
        'file_type': file_type,
        'num_sweeps': num_sweeps,
        'num_breaths': num_breaths or 0
    })


def log_feature_used(feature_name):
    """
    Log that a feature was used.

    Args:
        feature_name (str): Feature identifier
            Examples: 'gmm_clustering', 'manual_editing_add_peak',
                     'spectral_analysis', 'mark_sniff', 'move_point'
    """
    global _session_data
    _session_data['features_used'].add(feature_name)

    log_event('feature_used', {'feature': feature_name})


def log_export(export_type):
    """
    Log that data was exported.

    Args:
        export_type (str): Export type identifier
            Examples: 'summary_pdf', 'breaths_csv', 'timeseries_csv',
                     'events_csv', 'npz_session'
    """
    global _session_data
    _session_data['exports'][export_type] = \
        _session_data['exports'].get(export_type, 0) + 1

    log_event('export', {'export_type': export_type})


def log_session_end():
    """
    Log session summary when app closes.

    Call this in the app's closeEvent.
    """
    if not is_telemetry_enabled():
        return

    try:
        # Calculate session duration
        if _session_data['session_start']:
            start = datetime.fromisoformat(_session_data['session_start'])
            duration_minutes = (datetime.now() - start).total_seconds() / 60
        else:
            duration_minutes = 0

        # Send session summary to GA4
        log_event('session_end', {
            'session_duration_minutes': round(duration_minutes, 1),
            'files_analyzed': _session_data['files_analyzed'],
            'file_types_abf': _session_data['file_types'].get('abf', 0),
            'file_types_smrx': _session_data['file_types'].get('smrx', 0),
            'file_types_edf': _session_data['file_types'].get('edf', 0),
            'total_breaths': _session_data['total_breaths'],
            'total_sweeps': _session_data['total_sweeps'],
            'features_used_count': len(_session_data['features_used']),
            'exports_count': sum(_session_data['exports'].values())
        })

    except Exception as e:
        print(f"Warning: Could not log session end: {e}")


# ============================================================================
# Crash Reports (sent to Sentry)
# ============================================================================

def log_error(error, context=None):
    """
    Log an error or exception to Sentry.

    Args:
        error (Exception): Exception object
        context (dict, optional): Additional context (no PII)
    """
    if not is_crash_reports_enabled():
        return

    try:
        import sentry_sdk

        # Add context if provided
        if context:
            sentry_sdk.set_context("error_context", context)

        # Send to Sentry
        sentry_sdk.capture_exception(error)

    except ImportError:
        # Sentry not installed - log locally
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context
        }
        _log_event_locally({
            'event': 'error',
            'timestamp': datetime.now().isoformat(),
            'user_id': get_user_id(),
            'version': VERSION_STRING,
            'data': error_data
        })
    except Exception as e:
        # Silently fail
        print(f"Warning: Could not log error to Sentry: {e}")


# ============================================================================
# Utility
# ============================================================================

def is_active():
    """Check if telemetry is initialized and enabled."""
    return _telemetry_initialized and is_telemetry_enabled()
