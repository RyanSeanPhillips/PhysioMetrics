"""
Anonymous usage tracking and telemetry for PhysioMetrics.

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
    'edits_made': 0,  # Total manual edits (add/delete/move) - session-wide
    'edits_added': 0,  # False negatives (missed breaths) - session-wide
    'edits_deleted': 0,  # False positives (wrong detections) - session-wide
    'last_action': None,  # Last button/feature used (for crash tracking)
    'timing_data': {},  # Track operation durations
    # Per-file tracking (for ML evaluation - only saved files)
    'current_file_edits_added': 0,  # Edits added for current file
    'current_file_edits_deleted': 0,  # Edits deleted for current file
    'current_file_breaths': 0,  # Total breaths in current file (when detected)
    # Engagement tracking (for GA4 Realtime active users)
    'last_engagement_time': None,  # Last time user interacted with app
    'total_engagement_time_ms': 0,  # Cumulative engagement time
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

        # Sync any cached events from previous sessions (when network was down)
        sync_cached_events()

        # Send session start event to GA4
        log_event('session_start', {
            'version': VERSION_STRING,
            'platform': sys.platform,
            'python_version': platform.python_version()
        })

        # Send initial user_engagement event to help GA4 recognize active user
        log_user_engagement()

        print("Telemetry: Initialized (GA4 + Sentry)")

    except Exception as e:
        print(f"Warning: Could not initialize telemetry: {e}")


def _update_engagement_time():
    """
    Update engagement time tracking and return time since last engagement.

    This helps GA4 recognize active users in Realtime reports.

    Returns:
        int: Milliseconds since last engagement (0 if first engagement)
    """
    global _session_data
    import time

    current_time = time.time()
    last_time = _session_data.get('last_engagement_time')

    if last_time is None:
        # First engagement
        _session_data['last_engagement_time'] = current_time
        return 0

    # Calculate time since last engagement (cap at 60 seconds for reasonable values)
    elapsed_ms = int((current_time - last_time) * 1000)
    elapsed_ms = min(elapsed_ms, 60000)  # Cap at 60 seconds

    # Update tracking
    _session_data['last_engagement_time'] = current_time
    _session_data['total_engagement_time_ms'] += elapsed_ms

    return elapsed_ms


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

        # Log success (helps with debugging)
        if response.status_code in (200, 204):
            print(f"Telemetry: Sent '{event_name}' to GA4 (status {response.status_code})")

    except ImportError:
        # requests not available - log locally
        print("Telemetry: 'requests' module not available, caching locally")
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


def sync_cached_events():
    """
    Upload any cached events from telemetry.log to GA4.

    This is called on app startup to sync events that were cached
    when the network was unavailable. Events are sent in background
    threads and the log file is deleted after successful upload.
    """
    if not GA4_MEASUREMENT_ID or not GA4_API_SECRET:
        return  # Can't sync without GA4 credentials

    try:
        log_file = get_config_dir() / 'telemetry.log'

        if not log_file.exists():
            return  # No cached events

        # Read all cached events
        with open(log_file, 'r') as f:
            lines = f.readlines()

        if not lines:
            log_file.unlink()  # Delete empty file
            return

        # Send each event to GA4
        uploaded = 0
        for line in lines:
            try:
                event_data = json.loads(line.strip())

                # Extract event name and params
                event_name = event_data.get('event')
                params = event_data.get('params', {})

                if event_name:
                    # Send to GA4 in background (non-blocking)
                    _send_to_google_analytics(event_name, params)
                    uploaded += 1

            except Exception:
                # Skip malformed lines
                continue

        # Delete log file after sending all events
        log_file.unlink()

        if uploaded > 0:
            print(f"Telemetry: Synced {uploaded} cached event(s) to GA4")

    except Exception as e:
        # Silently fail - don't interrupt startup
        print(f"Telemetry: Could not sync cached events: {e}")


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

        # Calculate and add engagement time (helps GA4 count active users)
        engagement_time_ms = _update_engagement_time()
        if engagement_time_ms > 0:
            params['engagement_time_msec'] = engagement_time_ms

        # Add version and platform (sent with EVERY event for filtering)
        params['app_version'] = VERSION_STRING  # Changed from 'version' for clarity
        params['platform'] = sys.platform
        params['python_version'] = platform.python_version()

        # Send to Google Analytics
        _send_to_google_analytics(event_name, params)

    except Exception as e:
        # Never crash app due to telemetry
        print(f"Warning: Telemetry event failed: {e}")


def log_file_loaded(file_type, num_sweeps, num_breaths=None, **extra_params):
    """
    Log that a file was loaded.

    Args:
        file_type (str): File extension ('abf', 'smrx', 'edf')
        num_sweeps (int): Number of sweeps in file
        num_breaths (int, optional): Number of breaths detected
        **extra_params: Additional file metadata
            Examples: file_size_mb, sampling_rate_hz, duration_minutes,
                     num_channels, selected_channel

    Example:
        log_file_loaded('abf', num_sweeps=10, num_breaths=247,
                       file_size_mb=15.2, sampling_rate_hz=1000,
                       duration_minutes=30, num_channels=4)
    """
    global _session_data

    _session_data['files_analyzed'] += 1
    _session_data['file_types'][file_type.lower()] = \
        _session_data['file_types'].get(file_type.lower(), 0) + 1
    _session_data['total_sweeps'] += num_sweeps

    if num_breaths:
        _session_data['total_breaths'] += num_breaths

    # Reset per-file edit tracking for new file
    _session_data['current_file_edits_added'] = 0
    _session_data['current_file_edits_deleted'] = 0
    _session_data['current_file_breaths'] = 0  # Will be set after peak detection

    params = {
        'file_type': file_type,
        'num_sweeps': num_sweeps,
        'num_breaths': num_breaths or 0
    }
    params.update(extra_params)

    log_event('file_loaded', params)


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


def log_file_saved(save_type='npz', eupnea_count=None, sniff_count=None, **extra_params):
    """
    Log file save/export with per-file edit metrics for ML evaluation.

    This is the KEY metric for measuring ML performance improvements:
    - Only tracks files that were completed and saved by the user
    - Excludes test files, abandoned analyses, and exploratory work
    - Directly comparable across app versions

    Args:
        save_type (str): Type of save ('npz', 'csv', 'pdf', 'consolidated')
        eupnea_count (int, optional): Number of breaths classified as eupnea
        sniff_count (int, optional): Number of breaths classified as sniffing
        **extra_params: Additional context (file_type, num_sweeps, etc.)

    Example:
        log_file_saved('npz', eupnea_count=180, sniff_count=20,
                      file_type='abf', num_sweeps=10)
    """
    global _session_data

    # Get per-file metrics
    num_breaths = _session_data['current_file_breaths']
    edits_added = _session_data['current_file_edits_added']
    edits_deleted = _session_data['current_file_edits_deleted']
    edits_total = edits_added + edits_deleted

    # Calculate per-file edit metrics (for ML evaluation)
    if num_breaths > 0:
        edit_percentage = (edits_total / num_breaths) * 100
        false_negative_rate = (edits_added / num_breaths) * 100
        false_positive_rate = (edits_deleted / num_breaths) * 100
    else:
        edit_percentage = 0
        false_negative_rate = 0
        false_positive_rate = 0

    # Calculate eupnea/sniffing metrics (for breath pattern analysis)
    eupnea_percentage = 0
    sniff_percentage = 0
    eupnea_to_sniff_ratio = 0

    if eupnea_count is not None and sniff_count is not None:
        total_classified = eupnea_count + sniff_count
        if total_classified > 0:
            eupnea_percentage = (eupnea_count / total_classified) * 100
            sniff_percentage = (sniff_count / total_classified) * 100
        if sniff_count > 0:
            eupnea_to_sniff_ratio = eupnea_count / sniff_count

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

    # Add eupnea/sniffing metrics if available
    if eupnea_count is not None and sniff_count is not None:
        params['eupnea_count'] = eupnea_count
        params['sniff_count'] = sniff_count
        params['eupnea_percentage'] = round(eupnea_percentage, 2)
        params['sniff_percentage'] = round(sniff_percentage, 2)
        params['eupnea_to_sniff_ratio'] = round(eupnea_to_sniff_ratio, 2)

    params.update(extra_params)

    log_event('file_saved', params)

    print(f"[telemetry] File saved with {edit_percentage:.1f}% edits "
          f"({edits_added} added, {edits_deleted} deleted of {num_breaths} breaths)")


def log_user_engagement():
    """
    Send a user_engagement event to GA4.

    This is a special GA4 event that helps count active users in Realtime reports.
    Call this periodically (e.g., every 30-60 seconds) while app is in use.

    Can also be called on any user interaction to signal engagement.
    """
    if not is_telemetry_enabled():
        return

    # engagement_time_msec is automatically added by log_event via _update_engagement_time
    log_event('user_engagement', {})


def log_screen_view(screen_name, screen_class=None, **extra_params):
    """
    Log when user views a screen/dialog.

    This is the desktop app equivalent of "page views" for websites.
    Shows which screens users spend time in.

    Args:
        screen_name (str): Name of the screen/dialog (e.g., 'GMM Clustering Dialog')
        screen_class (str, optional): Class name for grouping (e.g., 'dialog', 'main_screen')
        **extra_params: Additional context

    Example:
        log_screen_view('GMM Clustering Dialog', screen_class='analysis_dialog')
        log_screen_view('Main Analysis Screen')
    """
    if not is_telemetry_enabled():
        return

    params = {
        'screen_name': screen_name,
        'firebase_screen': screen_name,  # GA4 standard parameter for screen name
    }

    if screen_class:
        params['screen_class'] = screen_class
        params['firebase_screen_class'] = screen_class  # GA4 standard parameter

    params.update(extra_params)

    log_event('screen_view', params)


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

        # Calculate edit metrics for ML performance tracking
        total_breaths = _session_data['total_breaths']
        edits_made = _session_data['edits_made']
        edits_added = _session_data['edits_added']
        edits_deleted = _session_data['edits_deleted']

        # Edit percentage (% of breaths that needed manual correction)
        edit_percentage = (edits_made / total_breaths * 100) if total_breaths > 0 else 0

        # False negative rate (missed breaths / total)
        false_negative_rate = (edits_added / total_breaths * 100) if total_breaths > 0 else 0

        # False positive rate (wrong detections / total)
        false_positive_rate = (edits_deleted / total_breaths * 100) if total_breaths > 0 else 0

        # Edits per file (average correction burden per file)
        edits_per_file = (edits_made / _session_data['files_analyzed']) if _session_data['files_analyzed'] > 0 else 0

        # Send session summary to GA4
        log_event('session_end', {
            'session_duration_minutes': round(duration_minutes, 1),
            'files_analyzed': _session_data['files_analyzed'],
            'file_types_abf': _session_data['file_types'].get('abf', 0),
            'file_types_smrx': _session_data['file_types'].get('smrx', 0),
            'file_types_edf': _session_data['file_types'].get('edf', 0),
            'total_breaths': total_breaths,
            'total_sweeps': _session_data['total_sweeps'],
            'features_used_count': len(_session_data['features_used']),
            'exports_count': sum(_session_data['exports'].values()),
            'edits_made': edits_made,
            'edits_added': edits_added,
            'edits_deleted': edits_deleted,
            'edit_percentage': round(edit_percentage, 2),
            'false_negative_rate': round(false_negative_rate, 2),
            'false_positive_rate': round(false_positive_rate, 2),
            'edits_per_file': round(edits_per_file, 2)
        })

    except Exception as e:
        print(f"Warning: Could not log session end: {e}")


# ============================================================================
# Enhanced Tracking - Timing, Edits, and Detailed Usage
# ============================================================================

def log_timing(operation_name, duration_seconds, **extra_params):
    """
    Log timing data for operations.

    Args:
        operation_name (str): Name of operation
            Examples: 'peak_detection', 'file_load', 'gmm_clustering',
                     'channel_selection_to_save'
        duration_seconds (float): Duration in seconds
        **extra_params: Additional parameters (e.g., num_breaths, file_size)

    Example:
        log_timing('peak_detection', 2.5, num_breaths=247, file_size_mb=15)
    """
    global _session_data

    # Store in session data for aggregation
    if operation_name not in _session_data['timing_data']:
        _session_data['timing_data'][operation_name] = []
    _session_data['timing_data'][operation_name].append(duration_seconds)

    # Send to GA4
    params = {
        'operation': operation_name,
        'duration_seconds': round(duration_seconds, 2)
    }
    params.update(extra_params)

    log_event('timing', params)


def log_edit(edit_type, **extra_params):
    """
    Log manual editing actions.

    Args:
        edit_type (str): Type of edit
            Examples: 'add_peak', 'delete_peak', 'move_peak', 'mark_sniff'
        **extra_params: Additional context (e.g., num_peaks_remaining)

    Example:
        log_edit('add_peak', num_peaks_after=248)
    """
    global _session_data

    # Track session-wide edits
    _session_data['edits_made'] += 1
    _session_data['last_action'] = f'edit_{edit_type}'

    # Track add vs delete separately for ML performance metrics (session-wide)
    if edit_type == 'add_peak':
        _session_data['edits_added'] += 1
        # Also track per-file (for ML evaluation of saved files)
        _session_data['current_file_edits_added'] += 1
    elif edit_type == 'delete_peak':
        _session_data['edits_deleted'] += 1
        # Also track per-file (for ML evaluation of saved files)
        _session_data['current_file_edits_deleted'] += 1

    params = {'edit_type': edit_type}
    params.update(extra_params)

    log_event('manual_edit', params)


def log_button_click(button_name, **extra_params):
    """
    Log button/UI interactions.

    Args:
        button_name (str): Button/action identifier
            Examples: 'detect_peaks', 'apply_filter', 'run_gmm',
                     'export_csv', 'save_session', 'load_session'
        **extra_params: Additional context

    Example:
        log_button_click('detect_peaks', threshold=0.5)
    """
    global _session_data

    _session_data['last_action'] = button_name

    params = {'button': button_name}
    params.update(extra_params)

    log_event('button_click', params)


def log_breath_statistics(num_breaths, mean_frequency=None, regularity_score=None, **extra_params):
    """
    Log breathing analysis statistics.

    Args:
        num_breaths (int): Total breaths detected
        mean_frequency (float, optional): Mean breathing frequency (Hz)
        regularity_score (float, optional): Breathing regularity score
        **extra_params: Additional metrics

    Example:
        log_breath_statistics(247, mean_frequency=1.2, regularity_score=0.85,
                            eupnea_percentage=75, apnea_count=3)
    """
    params = {
        'num_breaths': num_breaths
    }

    if mean_frequency is not None:
        params['mean_frequency_hz'] = round(mean_frequency, 2)

    if regularity_score is not None:
        params['regularity_score'] = round(regularity_score, 3)

    params.update(extra_params)

    log_event('breath_statistics', params)


def log_crash(error_message, **extra_params):
    """
    Log application crash/error to GA4.

    Args:
        error_message (str): Brief error description (no PII)
        **extra_params: Additional context

    Example:
        log_crash('IndexError in peak detection',
                 last_action=_session_data.get('last_action'),
                 num_breaths=247)
    """
    global _session_data

    params = {
        'error_type': error_message,
        'last_action': _session_data.get('last_action', 'unknown')
    }
    params.update(extra_params)

    log_event('crash', params)

    # Also try to send to Sentry if enabled
    if is_crash_reports_enabled():
        try:
            import sentry_sdk
            sentry_sdk.capture_message(
                f"App crash: {error_message}",
                level="error",
                contexts={"crash_context": params}
            )
        except:
            pass  # Sentry might not be working


def log_warning(warning_message, **extra_params):
    """
    Log non-critical warnings (e.g., "No peaks detected", "Filter unstable").

    Args:
        warning_message (str): Brief warning description
        **extra_params: Additional context

    Example:
        log_warning('No peaks detected', threshold=0.5, data_points=10000)
    """
    params = {'warning_type': warning_message}
    params.update(extra_params)

    log_event('warning', params)


def log_filter_applied(filter_type, **params):
    """
    Log filter application with settings.

    Args:
        filter_type (str): Type of filter ('butterworth', 'notch', 'mean_subtract')
        **params: Filter parameters

    Example:
        log_filter_applied('butterworth', highpass=0.5, lowpass=10.0,
                          order=4, data_points=100000)
    """
    telemetry_params = {'filter_type': filter_type}
    telemetry_params.update(params)

    log_event('filter_applied', telemetry_params)


def log_peak_detection(method, num_peaks, **params):
    """
    Log peak detection results.

    Args:
        method (str): Detection method ('auto_threshold', 'manual_threshold',
                                        'template_matching', 'derivative')
        num_peaks (int): Number of peaks detected
        **params: Detection parameters

    Example:
        log_peak_detection('auto_threshold', num_peaks=247,
                          threshold=0.42, min_distance=100)
    """
    global _session_data

    # Set current file breath count (for per-file edit percentage tracking)
    num_breaths = params.get('num_breaths', num_peaks)
    _session_data['current_file_breaths'] = num_breaths

    telemetry_params = {
        'detection_method': method,
        'num_peaks': num_peaks
    }
    telemetry_params.update(params)

    log_event('peak_detection', telemetry_params)


def log_navigation(action, **params):
    """
    Log navigation actions (sweep changes, window scrolling, zooming).

    Args:
        action (str): Navigation action ('change_sweep', 'zoom_in', 'zoom_out',
                                        'pan_left', 'pan_right', 'reset_view')
        **params: Additional context

    Example:
        log_navigation('change_sweep', sweep_number=5, total_sweeps=10)
    """
    telemetry_params = {'navigation_action': action}
    telemetry_params.update(params)

    log_event('navigation', telemetry_params)


def log_keyboard_shortcut(shortcut_name, **params):
    """
    Log keyboard shortcut usage.

    Args:
        shortcut_name (str): Shortcut identifier
            Examples: 'shift_click_add_peak', 'ctrl_click_delete_peak',
                     'f1_help', 'ctrl_s_save'
        **params: Additional context

    Example:
        log_keyboard_shortcut('shift_click_add_peak')
    """
    telemetry_params = {'shortcut': shortcut_name}
    telemetry_params.update(params)

    log_event('keyboard_shortcut', telemetry_params)


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
