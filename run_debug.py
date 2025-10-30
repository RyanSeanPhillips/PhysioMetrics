#!/usr/bin/env python3
"""
Debug launcher for PlethApp
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
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
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
    print("PlethApp Debug Launcher")
    print("="*50)

    # Debug: Show environment variables
    print(f"[DEBUG] PLETHAPP_TESTING = '{os.environ.get('PLETHAPP_TESTING', 'NOT SET')}'")

    setup_environment()

    if not check_imports():
        print("\nSome imports failed. The application may not work correctly.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

    print("\nLaunching PlethApp...")
    print("Close the application window to return to this prompt.")
    print("-" * 50)

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

        # Install global exception handler for crash tracking
        def exception_hook(exctype, value, tb):
            """Catch unhandled exceptions and log to telemetry."""
            import traceback

            # Log crash to telemetry
            telemetry.log_crash(
                error_message=f"{exctype.__name__}: {str(value)[:100]}",
                traceback_depth=len(traceback.extract_tb(tb))
            )

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
            f"Loading PlethApp v{VERSION_STRING}...",
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
            Qt.GlobalColor.white
        )
        splash.show()
        app.processEvents()

        # Create main window
        w = MainWindow()

        # Close splash and show main window
        splash.finish(w)
        w.show()

        sys.exit(app.exec())

    except Exception as e:
        print(f"\nERROR: Application crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nApplication closed normally.")

if __name__ == '__main__':
    main()