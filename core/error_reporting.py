"""
Error and crash reporting system for PhysioMetrics.

Provides:
- Local crash report persistence with full stack traces
- Rate-limited error logging for non-crash errors
- GitHub issue URL generation for easy bug reporting
- Session crash detection on startup

Crash reports are saved to: {config_dir}/crash_reports/
Error logs are saved to: {config_dir}/error_log.json
"""

import sys
import platform
import json
import traceback
import uuid
import atexit
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

from version_info import VERSION_STRING


# ============================================================================
# Constants
# ============================================================================

MAX_ERRORS_PER_SESSION = 100
CRASH_REPORTS_DIR = "crash_reports"
ERROR_LOG_FILE = "error_log.json"
SESSION_LOCK_FILE = ".session_lock"
HEARTBEAT_FILE = ".heartbeat"  # Tracks current app state for kill detection
DEBUG_LOG_FILE = "debug_log.json"  # Persistent action log (survives restart)
MAX_DEBUG_LOG_ENTRIES = 100  # Keep last N actions in debug log
GITHUB_REPO = "RyanSeanPhillips/PhysioMetrics"

# Maximum traceback length in GitHub issue body (URL length limits)
MAX_TRACEBACK_FOR_GITHUB = 1500

# Heartbeat update interval (seconds)
HEARTBEAT_INTERVAL_SEC = 5


# ============================================================================
# CrashReport Data Class
# ============================================================================

@dataclass
class CrashReport:
    """Structured crash report data."""

    # Metadata
    schema_version: str = "1.0"
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    submitted: bool = False

    # Error details
    error_type: str = ""
    error_message: str = ""
    traceback_str: str = ""
    traceback_depth: int = 0

    # Context from session
    last_action: str = "unknown"
    files_analyzed: int = 0
    total_breaths: int = 0
    features_used: List[str] = field(default_factory=list)
    edits_made: int = 0

    # System info
    app_version: str = VERSION_STRING
    platform_name: str = sys.platform
    python_version: str = platform.python_version()
    os_version: str = field(default_factory=lambda: platform.platform())

    # Session info
    session_start: str = ""
    user_id: str = ""

    # Heartbeat data (for killed sessions - raw JSON for debugging)
    heartbeat_json: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CrashReport':
        """Create from dictionary."""
        # Handle missing fields gracefully
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> 'CrashReport':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


# ============================================================================
# ErrorReporter Singleton
# ============================================================================

class ErrorReporter:
    """
    Central error reporting system.

    Responsibilities:
    - Save crash reports to local files
    - Log non-crash errors with rate limiting
    - Generate GitHub issue URLs
    - Detect crashes from previous sessions
    """

    _instance: Optional['ErrorReporter'] = None

    @classmethod
    def get_instance(cls) -> 'ErrorReporter':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Import here to avoid circular imports
        from core.config import get_config_dir

        self._config_dir = get_config_dir()
        self._crash_reports_dir = self._config_dir / CRASH_REPORTS_DIR
        self._crash_reports_dir.mkdir(exist_ok=True)

        self._session_id = str(uuid.uuid4())[:8]
        self._session_start = datetime.now()
        self._error_count = 0
        self._seen_errors: Dict[str, int] = {}  # error_key -> count
        self._session_lock_path = self._config_dir / SESSION_LOCK_FILE
        self._heartbeat_path = self._config_dir / HEARTBEAT_FILE
        self._debug_log_path = self._config_dir / DEBUG_LOG_FILE
        self._previous_crash_detected = False
        self._previous_session_info: Optional[Dict] = None  # Saved before overwriting lock

    # ========================================================================
    # Crash Report Management
    # ========================================================================

    def save_crash_report(
        self,
        exc_type: type,
        exc_value: Exception,
        exc_tb,
        session_data: Optional[Dict] = None
    ) -> Path:
        """
        Save a crash report to local file.

        Args:
            exc_type: Exception type
            exc_value: Exception instance
            exc_tb: Traceback object
            session_data: Optional context from telemetry session

        Returns:
            Path to saved crash report file
        """
        # Build traceback string
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        tb_str = "".join(tb_lines)

        # Create crash report
        report = CrashReport(
            error_type=exc_type.__name__,
            error_message=str(exc_value)[:500],  # Truncate long messages
            traceback_str=tb_str,
            traceback_depth=len(traceback.extract_tb(exc_tb)),
        )

        # Add session context if available
        if session_data:
            report.last_action = session_data.get('last_action', 'unknown')
            report.files_analyzed = session_data.get('files_analyzed', 0)
            report.total_breaths = session_data.get('total_breaths', 0)
            report.features_used = session_data.get('features_used', [])
            report.edits_made = session_data.get('edits_made', 0)
            report.session_start = session_data.get('session_start', '')

        # Include current heartbeat data (action history)
        if hasattr(self, '_action_history') and self._action_history:
            heartbeat_data = {
                "session_id": self._session_id,
                "timestamp": datetime.now().isoformat(),
                "current_action": report.last_action,
                "uptime_seconds": (datetime.now() - self._session_start).total_seconds(),
                "action_history": self._action_history
            }
            report.heartbeat_json = json.dumps(heartbeat_data, indent=2)

        # Add user ID from config
        try:
            from core.config import get_user_id
            report.user_id = get_user_id()
        except Exception:
            pass

        # Generate filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = report.report_id[:8]
        filename = f"crash_{timestamp_str}_{short_id}.json"

        # Save to file
        crash_path = self._crash_reports_dir / filename
        crash_path.write_text(report.to_json(), encoding='utf-8')

        print(f"[Crash Report] Saved to: {crash_path}")

        return crash_path

    def load_crash_report(self, path: Path) -> Optional[CrashReport]:
        """Load a crash report from file."""
        try:
            json_str = path.read_text(encoding='utf-8')
            return CrashReport.from_json(json_str)
        except Exception as e:
            print(f"[Crash Report] Failed to load {path}: {e}")
            return None

    def get_pending_crash_reports(self) -> List[Path]:
        """
        Get all unsubmitted crash reports.

        Returns:
            List of paths to crash report files where submitted=False
        """
        pending = []

        for crash_file in self._crash_reports_dir.glob("crash_*.json"):
            try:
                report = self.load_crash_report(crash_file)
                if report and not report.submitted:
                    pending.append(crash_file)
            except Exception:
                continue

        # Sort by modification time (newest first)
        pending.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        return pending

    def get_most_recent_pending_report(self) -> Optional[CrashReport]:
        """Get the most recent unsubmitted crash report."""
        pending = self.get_pending_crash_reports()
        if pending:
            return self.load_crash_report(pending[0])
        return None

    def mark_crash_report_submitted(self, report_id: str):
        """Mark a crash report as submitted."""
        for crash_file in self._crash_reports_dir.glob("crash_*.json"):
            try:
                report = self.load_crash_report(crash_file)
                if report and report.report_id == report_id:
                    report.submitted = True
                    crash_file.write_text(report.to_json(), encoding='utf-8')
                    print(f"[Crash Report] Marked as submitted: {crash_file.name}")
                    return
            except Exception:
                continue

    def cleanup_old_crash_reports(self, max_age_days: int = 30, max_count: int = 20):
        """
        Remove old crash reports to prevent disk bloat.

        Args:
            max_age_days: Remove reports older than this
            max_count: Keep at most this many reports
        """
        import time

        crash_files = list(self._crash_reports_dir.glob("crash_*.json"))

        # Sort by modification time (oldest first)
        crash_files.sort(key=lambda p: p.stat().st_mtime)

        now = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60

        removed = 0

        for crash_file in crash_files:
            try:
                age = now - crash_file.stat().st_mtime

                # Remove if too old
                if age > max_age_seconds:
                    crash_file.unlink()
                    removed += 1
                    continue

                # Remove if we have too many (keep newest)
                remaining = len(crash_files) - removed
                if remaining > max_count:
                    crash_file.unlink()
                    removed += 1

            except Exception:
                continue

        if removed > 0:
            print(f"[Crash Report] Cleaned up {removed} old report(s)")

    # ========================================================================
    # Error Logging (Rate-Limited)
    # ========================================================================

    def log_error(
        self,
        error: Exception,
        context: Optional[str] = None,
        extra_data: Optional[Dict] = None
    ) -> bool:
        """
        Log a non-crash error with rate limiting.

        Rate limiting rules:
        - Max 100 errors per session total
        - Deduplicate identical errors (same type + message)

        Args:
            error: The exception to log
            context: Optional context string (e.g., "peak_detection")
            extra_data: Optional additional data

        Returns:
            True if logged, False if rate-limited
        """
        if self._error_count >= MAX_ERRORS_PER_SESSION:
            return False

        # Create error key for deduplication
        error_key = f"{type(error).__name__}:{str(error)[:100]}"

        if error_key in self._seen_errors:
            # Increment count but don't log again
            self._seen_errors[error_key] += 1
            return False

        self._seen_errors[error_key] = 1
        self._error_count += 1

        # Build error entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self._session_id,
            "error_type": type(error).__name__,
            "error_message": str(error)[:500],
            "context": context,
            "extra_data": extra_data,
            "traceback": traceback.format_exc()[:2000]  # Truncate long tracebacks
        }

        # Append to error log file
        self._append_to_error_log(entry)

        return True

    def _append_to_error_log(self, entry: Dict):
        """Append an error entry to the log file."""
        error_log_path = self._config_dir / ERROR_LOG_FILE

        try:
            # Load existing log
            if error_log_path.exists():
                try:
                    log_data = json.loads(error_log_path.read_text(encoding='utf-8'))
                    if not isinstance(log_data, list):
                        log_data = []
                except Exception:
                    log_data = []
            else:
                log_data = []

            # Append new entry
            log_data.append(entry)

            # Keep only last 500 entries to prevent bloat
            if len(log_data) > 500:
                log_data = log_data[-500:]

            # Save
            error_log_path.write_text(
                json.dumps(log_data, indent=2, default=str),
                encoding='utf-8'
            )

        except Exception as e:
            # Silently fail - never crash due to error logging
            print(f"[Error Log] Failed to write: {e}")

    def get_error_count(self) -> int:
        """Get number of errors logged this session."""
        return self._error_count

    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error types and counts."""
        return dict(self._seen_errors)

    # ========================================================================
    # GitHub Issue URL Generation
    # ========================================================================

    def generate_github_issue_url(
        self,
        crash_report: Optional[CrashReport] = None,
        title: Optional[str] = None,
        body: Optional[str] = None,
        user_description: Optional[str] = None
    ) -> str:
        """
        Generate a GitHub issue URL with pre-filled content.

        Args:
            crash_report: Optional CrashReport to format
            title: Optional custom title
            body: Optional custom body
            user_description: Optional user-provided description of what they were doing

        Returns:
            GitHub issue URL with query parameters
        """
        import urllib.parse

        base_url = f"https://github.com/{GITHUB_REPO}/issues/new"

        # Generate title and body from crash report
        if crash_report and not title:
            # Truncate error message for title
            error_msg = crash_report.error_message[:50]
            if len(crash_report.error_message) > 50:
                error_msg += "..."
            title = f"[Crash Report] {crash_report.error_type}: {error_msg}"

        if crash_report and not body:
            body = self._format_crash_report_for_issue(crash_report, user_description)

        # Build query parameters
        params = {
            "title": title or "Bug Report",
            "body": body or self._get_default_issue_body(),
            "labels": "bug"
        }

        # URL encode the parameters
        query_string = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)

        return f"{base_url}?{query_string}"

    def _format_crash_report_for_issue(self, report: CrashReport, user_description: Optional[str] = None) -> str:
        """Format a crash report as GitHub issue body (Markdown)."""
        # Truncate traceback for URL length limits
        traceback_truncated = report.traceback_str[:MAX_TRACEBACK_FOR_GITHUB]
        if len(report.traceback_str) > MAX_TRACEBACK_FOR_GITHUB:
            traceback_truncated += "\n... (truncated)"

        # Format user description section
        if user_description:
            user_section = f"""### User Description
{user_description}

"""
        else:
            user_section = """### Steps to Reproduce
<!-- Please describe what you were doing when this crash occurred -->

1.
2.
3.

"""

        # Include heartbeat data if available (collapsible section)
        if report.heartbeat_json:
            heartbeat_section = f"""
<details>
<summary>Heartbeat Data (click to expand)</summary>

```json
{report.heartbeat_json}
```
</details>

"""
        else:
            heartbeat_section = ""

        return f"""## Crash Report

**App Version:** {report.app_version}
**Platform:** {report.platform_name}
**Python:** {report.python_version}
**OS:** {report.os_version}

### Error
```
{report.error_type}: {report.error_message}
```

### Stack Trace
```python
{traceback_truncated}
```

### Context
- **Last Action:** {report.last_action}
- **Files Analyzed:** {report.files_analyzed}
- **Total Breaths:** {report.total_breaths}

{user_section}{heartbeat_section}---
*Report ID: {report.report_id[:8]}*
*Generated by PhysioMetrics crash reporter*
"""

    def _get_default_issue_body(self) -> str:
        """Get default issue body template."""
        return f"""## Bug Report

**App Version:** {VERSION_STRING}
**Platform:** {sys.platform}
**Python:** {platform.python_version()}

### Description
<!-- Describe the bug -->


### Steps to Reproduce
1.
2.
3.

### Expected Behavior
<!-- What should happen? -->


### Actual Behavior
<!-- What happens instead? -->


---
*Generated by PhysioMetrics*
"""

    # ========================================================================
    # Session Lock Management (for crash detection)
    # ========================================================================

    def write_session_lock(self):
        """
        Write a lock file indicating app is running.

        This file is removed on clean exit. If it exists on startup,
        the previous session crashed.
        """
        lock_data = {
            "session_id": self._session_id,
            "started": datetime.now().isoformat(),
            "pid": "unknown"
        }

        try:
            import os
            lock_data["pid"] = str(os.getpid())
        except Exception:
            pass

        try:
            self._session_lock_path.write_text(
                json.dumps(lock_data, indent=2),
                encoding='utf-8'
            )
            print(f"[Session] Lock written: {self._session_lock_path}")
        except Exception as e:
            print(f"[Session] Failed to write lock: {e}")

    def clear_session_lock(self):
        """Remove session lock file (called on clean exit)."""
        try:
            if self._session_lock_path.exists():
                self._session_lock_path.unlink()
                print("[Session] Lock cleared (clean exit)")
        except Exception as e:
            print(f"[Session] Failed to clear lock: {e}")

    def _check_for_previous_crash(self) -> bool:
        """
        Check if previous session ended unexpectedly.
        Called BEFORE writing new session lock.

        Saves previous session info AND heartbeat for later use.

        Returns:
            True if lock file exists from a different session
        """
        if not self._session_lock_path.exists():
            return False

        try:
            lock_data = json.loads(self._session_lock_path.read_text(encoding='utf-8'))

            # If session_id matches current session, not a crash (shouldn't happen on fresh start)
            if lock_data.get('session_id') == self._session_id:
                return False

            # Lock file exists with different session_id = previous crash
            # Save the previous session info before we overwrite it
            self._previous_session_info = lock_data

            # IMPORTANT: Also read heartbeat NOW before it gets overwritten
            # This is the only chance to get it from the crashed session
            if self._heartbeat_path.exists():
                try:
                    heartbeat_data = json.loads(self._heartbeat_path.read_text(encoding='utf-8'))
                    # Verify it's from the crashed session
                    if heartbeat_data.get('session_id') == lock_data.get('session_id'):
                        self._previous_heartbeat = heartbeat_data
                        print(f"[Session] Captured heartbeat: {heartbeat_data.get('current_action')}")
                    else:
                        self._previous_heartbeat = None
                except Exception:
                    self._previous_heartbeat = None
            else:
                self._previous_heartbeat = None

            return True

        except Exception:
            # If we can't read the lock, assume crash
            return True

    def was_previous_session_crashed(self) -> bool:
        """
        Check if previous session ended unexpectedly.
        Uses cached value from init (since lock file is overwritten).

        Returns:
            True if previous session crashed
        """
        return getattr(self, '_previous_crash_detected', False)

    def get_previous_session_info(self) -> Optional[Dict]:
        """Get info about the crashed previous session (cached before overwrite)."""
        return self._previous_session_info

    # ========================================================================
    # Heartbeat Tracking (for kill detection)
    # ========================================================================

    # Maximum number of actions to keep in history
    MAX_ACTION_HISTORY = 20

    def update_heartbeat(self, current_action: str, extra_info: Optional[Dict] = None):
        """
        Update heartbeat file with current app state.

        Call this periodically and when starting long operations.
        If app is killed, this file shows what was happening.

        Args:
            current_action: Description of current activity (e.g., "exporting_csv")
            extra_info: Optional additional context (e.g., {"file": "data.abf"})
        """
        # Initialize action history if needed
        if not hasattr(self, '_action_history'):
            self._action_history = []

        # Add new action to history
        action_entry = {
            "action": current_action,
            "time": datetime.now().isoformat(),
            "context": extra_info.get('context') if extra_info else None
        }
        self._action_history.append(action_entry)

        # Keep only the last N actions (ring buffer)
        if len(self._action_history) > self.MAX_ACTION_HISTORY:
            self._action_history = self._action_history[-self.MAX_ACTION_HISTORY:]

        heartbeat_data = {
            "session_id": self._session_id,
            "timestamp": datetime.now().isoformat(),
            "current_action": current_action,
            "extra_info": extra_info or {},
            "uptime_seconds": (datetime.now() - self._session_start).total_seconds(),
            "action_history": self._action_history
        }

        try:
            self._heartbeat_path.write_text(
                json.dumps(heartbeat_data, indent=2),
                encoding='utf-8'
            )
        except Exception:
            pass  # Never crash due to heartbeat

    def get_last_heartbeat(self) -> Optional[Dict]:
        """Get the last heartbeat data (from crashed session)."""
        if not self._heartbeat_path.exists():
            return None

        try:
            return json.loads(self._heartbeat_path.read_text(encoding='utf-8'))
        except Exception:
            return None

    def clear_heartbeat(self):
        """Clear heartbeat file on clean exit."""
        try:
            if self._heartbeat_path.exists():
                self._heartbeat_path.unlink()
        except Exception:
            pass

    # ========================================================================
    # Persistent Debug Log (survives app restart)
    # ========================================================================

    def append_to_debug_log(self, action: str, context: Optional[str] = None,
                            level: str = "info", extra_data: Optional[Dict] = None):
        """
        Append an entry to the persistent debug log.

        Unlike heartbeat, this log persists across app restarts and is useful
        for debugging issues that don't cause crashes.

        Args:
            action: Description of the action (e.g., "button:detect_peaks")
            context: Optional context string
            level: Log level ("info", "warning", "error", "debug")
            extra_data: Optional additional data dict
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self._session_id,
            "action": action,
            "context": context,
            "level": level,
            "extra_data": extra_data or {}
        }

        try:
            # Read existing log
            if self._debug_log_path.exists():
                log_data = json.loads(self._debug_log_path.read_text(encoding='utf-8'))
                entries = log_data.get('entries', [])
            else:
                entries = []

            # Append new entry
            entries.append(entry)

            # Keep only last N entries
            if len(entries) > MAX_DEBUG_LOG_ENTRIES:
                entries = entries[-MAX_DEBUG_LOG_ENTRIES:]

            # Write back
            log_data = {
                "last_updated": datetime.now().isoformat(),
                "total_entries": len(entries),
                "entries": entries
            }
            self._debug_log_path.write_text(
                json.dumps(log_data, indent=2),
                encoding='utf-8'
            )
        except Exception:
            pass  # Never crash due to debug log

    def get_debug_log(self, limit: int = None) -> List[Dict]:
        """
        Get entries from the persistent debug log.

        Args:
            limit: Maximum number of entries to return (most recent first).
                   None returns all entries.

        Returns:
            List of log entries, most recent first
        """
        try:
            if not self._debug_log_path.exists():
                return []

            log_data = json.loads(self._debug_log_path.read_text(encoding='utf-8'))
            entries = log_data.get('entries', [])

            # Return in reverse order (most recent first)
            entries = list(reversed(entries))

            if limit:
                entries = entries[:limit]

            return entries
        except Exception:
            return []

    def clear_debug_log(self):
        """Clear the persistent debug log."""
        try:
            if self._debug_log_path.exists():
                self._debug_log_path.unlink()
        except Exception:
            pass

    def get_debug_log_path(self) -> Path:
        """Get the path to the debug log file."""
        return self._debug_log_path

    # ========================================================================
    # Kill Detection and Placeholder Report Generation
    # ========================================================================

    def get_crash_report_for_session(self, session_start_time: str) -> Optional[CrashReport]:
        """
        Find a crash report that matches the given session.

        Args:
            session_start_time: ISO format timestamp of session start

        Returns:
            CrashReport if found from same session, None otherwise
        """
        try:
            session_start = datetime.fromisoformat(session_start_time)
        except (ValueError, TypeError):
            return None

        # Look for crash reports created AFTER session start
        for crash_file in sorted(
            self._crash_reports_dir.glob("crash_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True  # Newest first
        ):
            try:
                report = self.load_crash_report(crash_file)
                if report:
                    report_time = datetime.fromisoformat(report.timestamp)
                    # Report should be after session start and within reasonable time
                    if report_time >= session_start:
                        return report
            except Exception:
                continue

        return None

    def generate_killed_report(self, previous_session_info: Dict, heartbeat: Optional[Dict] = None) -> CrashReport:
        """
        Generate a placeholder crash report for when app was killed.

        Args:
            previous_session_info: Session lock data from killed session
            heartbeat: Last heartbeat data if available

        Returns:
            CrashReport with "forcibly terminated" info
        """
        # Determine what was happening from heartbeat
        if heartbeat:
            last_action = heartbeat.get('current_action', 'unknown')
            extra_info = heartbeat.get('extra_info', {})
            uptime = heartbeat.get('uptime_seconds', 0)
            action_history = heartbeat.get('action_history', [])

            action_detail = f"Last recorded activity: {last_action}"
            if extra_info:
                action_detail += f" ({extra_info})"
            action_detail += f"\nSession uptime: {uptime:.0f} seconds"

            # Format action history (most recent last)
            if action_history:
                action_detail += f"\n\nRecent actions ({len(action_history)} recorded):"
                for entry in action_history:
                    action = entry.get('action', '?')
                    time = entry.get('time', '')
                    # Extract just the time portion (HH:MM:SS)
                    if 'T' in time:
                        time = time.split('T')[1].split('.')[0]
                    context = entry.get('context')
                    if context:
                        action_detail += f"\n  [{time}] {action} - {context}"
                    else:
                        action_detail += f"\n  [{time}] {action}"
        else:
            last_action = "unknown"
            action_detail = "No heartbeat data available"

        report = CrashReport(
            error_type="ForciblyTerminated",
            error_message="App was forcibly terminated (killed by user, Task Manager, or system)",
            traceback_str=f"""The application did not exit cleanly.

Possible causes:
- User closed via Task Manager
- System killed the process (out of memory, etc.)
- Power loss or system crash
- App became unresponsive and was force-closed

{action_detail}

Session started: {previous_session_info.get('started', 'unknown')}
Session ID: {previous_session_info.get('session_id', 'unknown')}
""",
            traceback_depth=0,
            last_action=last_action,
            session_start=previous_session_info.get('started', ''),
            heartbeat_json=json.dumps(heartbeat, indent=2) if heartbeat else "",
        )

        # Save this generated report
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = report.report_id[:8]
        filename = f"crash_{timestamp_str}_{short_id}.json"
        crash_path = self._crash_reports_dir / filename
        crash_path.write_text(report.to_json(), encoding='utf-8')

        print(f"[Crash Report] Generated killed-session report: {crash_path}")

        return report

    def get_appropriate_crash_report(self) -> Optional[CrashReport]:
        """
        Get the appropriate crash report for the previous crashed session.

        This handles both:
        1. Real crashes (exception caught, report saved)
        2. Kills (no exception, generate placeholder from heartbeat)

        Returns:
            CrashReport if previous session crashed, None otherwise
        """
        if not self._previous_crash_detected:
            return None

        prev_info = self._previous_session_info
        if not prev_info:
            return None

        # Try to find a real crash report from that session
        session_start = prev_info.get('started')
        if session_start:
            real_report = self.get_crash_report_for_session(session_start)
            if real_report:
                return real_report

        # No crash report found - must have been killed
        # Use the cached heartbeat from _check_for_previous_crash
        # (it was captured before being overwritten by the new session)
        heartbeat = getattr(self, '_previous_heartbeat', None)

        return self.generate_killed_report(prev_info, heartbeat)


# ============================================================================
# Module-Level Convenience Functions
# ============================================================================

def save_crash_report(exc_type, exc_value, exc_tb, session_data=None) -> Path:
    """Save a crash report to local file."""
    return ErrorReporter.get_instance().save_crash_report(
        exc_type, exc_value, exc_tb, session_data
    )


def log_error(error: Exception, context: str = None, extra_data: dict = None) -> bool:
    """Log a non-crash error with rate limiting."""
    return ErrorReporter.get_instance().log_error(error, context, extra_data)


def generate_github_issue_url(crash_report=None, title=None, body=None, user_description=None) -> str:
    """Generate a GitHub issue URL with pre-filled content."""
    return ErrorReporter.get_instance().generate_github_issue_url(
        crash_report, title, body, user_description
    )


def get_pending_crash_report() -> Optional[CrashReport]:
    """Get the most recent unsubmitted crash report."""
    return ErrorReporter.get_instance().get_most_recent_pending_report()


def get_appropriate_crash_report() -> Optional[CrashReport]:
    """
    Get the appropriate crash report for the previous crashed session.

    This handles both real crashes and kills (generates placeholder).
    """
    return ErrorReporter.get_instance().get_appropriate_crash_report()


def mark_submitted(report_id: str):
    """Mark a crash report as submitted."""
    ErrorReporter.get_instance().mark_crash_report_submitted(report_id)


def update_heartbeat(current_action: str, extra_info: dict = None):
    """Update heartbeat with current app state."""
    ErrorReporter.get_instance().update_heartbeat(current_action, extra_info)


def append_to_debug_log(action: str, context: str = None, level: str = "info", extra_data: dict = None):
    """Append an entry to the persistent debug log."""
    ErrorReporter.get_instance().append_to_debug_log(action, context, level, extra_data)


def get_debug_log(limit: int = None) -> List[Dict]:
    """Get entries from the persistent debug log (most recent first)."""
    return ErrorReporter.get_instance().get_debug_log(limit)


def get_debug_log_path() -> Path:
    """Get the path to the debug log file."""
    return ErrorReporter.get_instance().get_debug_log_path()


def init_error_reporter():
    """
    Initialize the error reporter.

    Call this early in app startup. Sets up session lock and
    registers cleanup handler.

    IMPORTANT: Checks for previous crash BEFORE writing new session lock.
    """
    reporter = ErrorReporter.get_instance()

    # Check for previous crash BEFORE writing new lock (order matters!)
    reporter._previous_crash_detected = reporter._check_for_previous_crash()
    if reporter._previous_crash_detected:
        print("[Session] Previous session crash detected!")

    # NOW write new session lock (overwrites old one)
    reporter.write_session_lock()

    # Write initial heartbeat
    reporter.update_heartbeat("app_starting")

    # Register cleanup on exit (clear both lock and heartbeat)
    def cleanup():
        reporter.clear_session_lock()
        reporter.clear_heartbeat()

    atexit.register(cleanup)

    # Clean up old crash reports
    reporter.cleanup_old_crash_reports()

    return reporter


def was_previous_session_crashed() -> bool:
    """Check if previous session crashed."""
    return ErrorReporter.get_instance().was_previous_session_crashed()
