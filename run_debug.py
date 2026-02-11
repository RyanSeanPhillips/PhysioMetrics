#!/usr/bin/env python3
"""
Debug launcher for PhysioMetrics
Use this to test the application before building the executable
"""

import sys
import os
from pathlib import Path

def setup_environment():
    """Setup the environment for running the application."""
    # Add the current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))

    # Change to the application directory
    os.chdir(current_dir)

    # Set Windows AppUserModelID so app shows as "PhysioMetrics" in Task Manager
    # and groups correctly in taskbar (instead of showing as "Python")
    try:
        import ctypes
        # This ID should match what's used in the built executable
        app_id = "RyanPhillips.PhysioMetrics.App.1"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass  # Not on Windows or API not available

def check_imports():
    """Check if all required modules can be imported."""
    print("Checking imports...")

    imports = [
        ('PyQt6', 'PyQt6.QtWidgets'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('scipy', 'scipy'),
        ('core.state', 'core.state'),
        ('core.abf_io', 'core.abf_io'),
        ('core.filters', 'core.filters'),
        ('core.plotting', 'core.plotting'),
        ('core.peaks', 'core.peaks'),
        ('core.metrics', 'core.metrics'),
    ]

    failed_imports = []

    for name, module in imports:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError as e:
            print(f"  [FAIL] {name}: {e}")
            failed_imports.append(name)

    if failed_imports:
        print(f"\nWARNING: Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("All imports successful!")
        return True

def main():
    """Main debug launcher."""
    print("="*50)
    print("PhysioMetrics Debug Launcher")
    print("="*50)

    # Debug: Show environment variables
    print(f"[DEBUG] PLETHAPP_TESTING = '{os.environ.get('PLETHAPP_TESTING', 'NOT SET')}'")

    setup_environment()

    if not check_imports():
        print("\nSome imports failed. The application may not work correctly.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

    print("\nLaunching PhysioMetrics...")
    print("Close the application window to return to this prompt.")
    print("-" * 50)

    # Set up early error handling BEFORE importing main app
    # This catches import errors that happen during module loading
    def save_early_crash(exc_type, exc_value, exc_tb):
        """Save crash report for errors that happen before full error handling is set up."""
        import traceback
        import json
        from datetime import datetime
        from pathlib import Path
        import uuid

        try:
            # Get config dir (minimal import)
            if sys.platform == 'win32':
                config_dir = Path(os.environ.get('APPDATA', '')) / 'PhysioMetrics'
            else:
                config_dir = Path.home() / '.config' / 'PhysioMetrics'

            crash_dir = config_dir / 'crash_reports'
            crash_dir.mkdir(parents=True, exist_ok=True)

            # Build minimal crash report
            tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            report = {
                "schema_version": "1.0",
                "report_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "submitted": False,
                "error_type": exc_type.__name__,
                "error_message": str(exc_value)[:500],
                "traceback_str": tb_str,
                "traceback_depth": len(traceback.extract_tb(exc_tb)),
                "last_action": "app_startup_import",
                "files_analyzed": 0,
                "total_breaths": 0,
                "features_used": [],
                "edits_made": 0,
                "app_version": "unknown",
                "platform_name": sys.platform,
                "python_version": sys.version.split()[0],
                "os_version": "",
                "session_start": datetime.now().isoformat(),
                "user_id": "",
                "heartbeat_json": ""
            }

            # Try to get version
            try:
                from version_info import VERSION_STRING
                report["app_version"] = VERSION_STRING
            except Exception:
                pass

            # Save crash report
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            short_id = report["report_id"][:8]
            filename = f"crash_{timestamp_str}_{short_id}.json"
            crash_path = crash_dir / filename
            crash_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

            print(f"\n[Early Crash] Saved crash report: {crash_path}")

        except Exception as e:
            print(f"[Early Crash] Failed to save crash report: {e}")

        # Call original handler
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    # Install early crash handler
    sys.excepthook = save_early_crash

    try:
        # Import and run the main application
        from PyQt6.QtWidgets import QApplication, QSplashScreen
        from PyQt6.QtGui import QPixmap
        from PyQt6.QtCore import Qt
        from main import MainWindow
        from version_info import VERSION_STRING

        app = QApplication(sys.argv)

        # Check for first launch and show welcome dialog
        from core import config as app_config
        if app_config.is_first_launch():
            from dialogs import FirstLaunchDialog
            first_launch_dialog = FirstLaunchDialog()
            if first_launch_dialog.exec():
                # User clicked Continue - save preferences
                telemetry_enabled, crash_reports_enabled = first_launch_dialog.get_preferences()
                cfg = app_config.load_config()
                cfg['telemetry_enabled'] = telemetry_enabled
                cfg['crash_reports_enabled'] = crash_reports_enabled
                cfg['first_launch'] = False
                app_config.save_config(cfg)
            else:
                # User closed dialog - use defaults and continue
                app_config.mark_first_launch_complete()

        # Initialize telemetry (after first-launch dialog)
        from core import telemetry
        telemetry.init_telemetry()

        # Initialize error reporter (writes session lock, registers cleanup)
        from core import error_reporting
        error_reporter = error_reporting.init_error_reporter()

        # Check if previous session crashed (before installing new hook)
        previous_crash = error_reporting.was_previous_session_crashed()
        # Use get_appropriate_crash_report which handles both real crashes and kills
        pending_report = error_reporting.get_appropriate_crash_report() if previous_crash else None

        # Install global exception handler for crash tracking
        def exception_hook(exctype, value, tb):
            """Catch unhandled exceptions, save report, and optionally show dialog."""
            import traceback

            # Log crash to GA4 telemetry (existing behavior)
            telemetry.log_crash(
                error_message=f"{exctype.__name__}: {str(value)[:100]}",
                traceback_depth=len(traceback.extract_tb(tb))
            )

            # Save crash report locally (new)
            if app_config.is_crash_reports_enabled():
                try:
                    session_data = telemetry.get_session_data()
                    crash_path = error_reporting.save_crash_report(
                        exctype, value, tb, session_data
                    )

                    # Try to show crash report dialog
                    # Only if QApplication is still running and stable
                    try:
                        from PyQt6.QtWidgets import QApplication
                        app_instance = QApplication.instance()
                        if app_instance:
                            from dialogs.crash_report_dialog import CrashReportDialog
                            report = error_reporting.ErrorReporter.get_instance().load_crash_report(crash_path)
                            if report:
                                dialog = CrashReportDialog(report, on_startup=False)
                                dialog.exec()
                    except Exception as dialog_error:
                        print(f"[Crash Report] Could not show dialog: {dialog_error}")

                except Exception as save_error:
                    print(f"[Crash Report] Failed to save: {save_error}")

            # Call default handler to print traceback
            sys.__excepthook__(exctype, value, tb)

        sys.excepthook = exception_hook

        # Create splash screen
        splash_paths = [
            Path(__file__).parent / "images" / "plethapp_splash_dark-01.png",
            Path(__file__).parent / "images" / "plethapp_splash.png",
            Path(__file__).parent / "images" / "plethapp_thumbnail_dark_round.ico",
            Path(__file__).parent / "assets" / "plethapp_thumbnail_dark_round.ico",
        ]

        splash_pix = None
        for splash_path in splash_paths:
            if splash_path.exists():
                splash_pix = QPixmap(str(splash_path))
                break

        if splash_pix is None or splash_pix.isNull():
            splash_pix = QPixmap(200, 150)
            splash_pix.fill(Qt.GlobalColor.darkGray)

        splash_pix = splash_pix.scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        splash = QSplashScreen(splash_pix, Qt.WindowType.WindowStaysOnTopHint)
        splash.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        splash.showMessage(
            f"Loading PhysioMetrics v{VERSION_STRING}...",
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
            Qt.GlobalColor.white
        )
        splash.show()
        app.processEvents()

        # Create main window
        w = MainWindow()

        # Add test crash shortcuts for debugging crash reporter
        from PyQt6.QtGui import QShortcut, QKeySequence
        import os as _os

        # Ctrl+Shift+T: Soft crash (exception, shows dialog immediately)
        def trigger_test_crash():
            """Intentionally raise an exception to test crash reporting."""
            print("[TEST] Triggering soft test crash (exception)...")
            # Simulate a realistic crash scenario
            test_data = {'peaks': [1, 2, 3], 'onsets': None}
            # This will raise TypeError: 'NoneType' object is not subscriptable
            _ = test_data['onsets'][0]

        test_crash_shortcut = QShortcut(QKeySequence("Ctrl+Shift+T"), w)
        test_crash_shortcut.activated.connect(trigger_test_crash)

        # Ctrl+Shift+K: Hard crash (kills app, dialog shows on NEXT startup)
        def trigger_hard_crash():
            """Abruptly terminate the app to test startup crash detection."""
            print("[TEST] Triggering HARD crash (app will terminate)...")
            print("[TEST] Restart the app to see 'Previous Session Crashed' dialog")
            # Save a crash report first so there's something to show
            try:
                session_data = telemetry.get_session_data()
                error_reporting.save_crash_report(
                    RuntimeError,
                    RuntimeError("Simulated hard crash for testing (Ctrl+Shift+K)"),
                    None,  # No traceback for simulated crash
                    session_data
                )
            except Exception as e:
                print(f"[TEST] Could not save crash report: {e}")
            # Force exit WITHOUT clearing session lock (simulates hard crash)
            _os._exit(1)

        hard_crash_shortcut = QShortcut(QKeySequence("Ctrl+Shift+K"), w)
        hard_crash_shortcut.activated.connect(trigger_hard_crash)

        print("[DEBUG] Test crash shortcuts enabled:")
        print("        Ctrl+Shift+T = Soft crash (exception, dialog now)")
        print("        Ctrl+Shift+K = Hard crash (kill app, dialog on restart)")

        # Close splash and show main window
        splash.finish(w)
        w.show()

        # Check for previous crash and show dialog (after window is visible)
        if pending_report and app_config.is_crash_reports_enabled():
            from PyQt6.QtCore import QTimer
            from dialogs.crash_report_dialog import show_crash_report_dialog

            def show_previous_crash_dialog():
                show_crash_report_dialog(pending_report, on_startup=True, parent=w)

            # Delay to let the main window fully initialize
            QTimer.singleShot(500, show_previous_crash_dialog)

        sys.exit(app.exec())

    except Exception as e:
        print(f"\nERROR: Application crashed: {e}")
        import traceback
        traceback.print_exc()

        # Save crash report using early crash handler
        exc_info = sys.exc_info()
        save_early_crash(exc_info[0], exc_info[1], exc_info[2])

        sys.exit(1)

    print("\nApplication closed normally.")

if __name__ == '__main__':
    main()