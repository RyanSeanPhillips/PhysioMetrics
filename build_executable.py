#!/usr/bin/env python3
"""
PlethApp Build Script - Python version
Creates Windows Executable using PyInstaller

This script provides a cross-platform way to build the executable
and includes additional validation and optimization options.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def clean_build_artifacts():
    """Clean build artifacts but preserve versioned dist/ folders."""
    print("Cleaning build artifacts...")

    # Only clean the temporary build directory and __pycache__
    # Keep dist/ folder to preserve version history
    directories_to_clean = ['build', '__pycache__']
    for dir_name in directories_to_clean:
        if os.path.exists(dir_name):
            try:
                shutil.rmtree(dir_name)
                print(f"  Removed {dir_name}/")
            except PermissionError:
                print(f"  Warning: Could not remove {dir_name}/ (files in use)")
                print(f"  PyInstaller will clean it automatically...")
            except Exception as e:
                print(f"  Warning: Error removing {dir_name}/: {e}")

    # Note: dist/ folder is NOT cleaned to preserve version history
    if os.path.exists('dist'):
        print(f"  Keeping dist/ folder (contains previous versions)")

    # Clean Python cache files recursively
    for root, dirs, files in os.walk('.'):
        # Remove __pycache__ directories
        for d in dirs[:]:  # Use slice to safely modify during iteration
            if d == '__pycache__':
                shutil.rmtree(os.path.join(root, d))
                dirs.remove(d)

        # Remove .pyc files
        for f in files:
            if f.endswith('.pyc'):
                os.remove(os.path.join(root, f))

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")

    required_modules = ['PyInstaller', 'PyQt6', 'numpy', 'pandas', 'matplotlib']
    missing_modules = []

    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"  ✗ {module} - MISSING")

    if missing_modules:
        print(f"\nERROR: Missing required modules: {', '.join(missing_modules)}")
        print("Please install them with:")
        print(f"pip install {' '.join(missing_modules)}")
        return False

    return True

def check_required_files():
    """Check if required files exist."""
    print("Checking required files...")

    required_files = [
        'main.py',
        'pleth_app.spec',
        'ui/pleth_app_layout.ui',  # At least one UI file
        'images/plethapp_thumbnail.ico',  # Application icon
    ]

    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"  ✗ {file_path} - MISSING")

    if missing_files:
        print(f"\nWARNING: Missing files: {', '.join(missing_files)}")
        print("Build may fail or executable may not work properly.")
        return False

    return True

def build_executable():
    """Build the executable using PyInstaller."""
    print("\nStarting PyInstaller build...")
    print("This may take several minutes...")

    # Get version string for output path
    try:
        from version_info import VERSION_STRING
        version = VERSION_STRING
    except ImportError:
        version = "unknown"

    try:
        # Run PyInstaller with the spec file
        result = subprocess.run([
            sys.executable, '-m', 'PyInstaller',
            '--clean',
            'pleth_app.spec'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("\n" + "="*60)
            print("BUILD SUCCESSFUL!")
            print("="*60)

            # Check for versioned directory output
            dist_dir = Path(f'dist/PlethApp_v{version}')
            exe_path = dist_dir / f'PlethApp_v{version}.exe'

            if exe_path.exists():
                file_size = exe_path.stat().st_size
                print(f"\nExecutable created: {exe_path}")
                print(f"File size: {file_size:,} bytes ({file_size / (1024*1024):.1f} MB)")
                print(f"\nBuild directory: {dist_dir}")
                print("\nYou can now distribute the entire folder to users.")
                print("The executable is self-contained and doesn't require Python installation.")

                # Create zip file for distribution
                print("\nCreating zip file for distribution...")
                zip_path = Path(f'dist/PlethApp_v{version}_Windows')
                try:
                    shutil.make_archive(str(zip_path), 'zip', 'dist', f'PlethApp_v{version}')
                    zip_file = Path(f'{zip_path}.zip')
                    if zip_file.exists():
                        zip_size = zip_file.stat().st_size
                        print("\nZIP FILE CREATED!")
                        print(f"Zip size: {zip_size:,} bytes ({zip_size / (1024*1024):.1f} MB)")
                        print(f"Location: {zip_file}")
                        print("\nReady to upload to GitHub releases!")
                    else:
                        print("\nWarning: Zip file was not created")
                except Exception as e:
                    print(f"\nWarning: Failed to create zip file: {e}")
                    print(f"You can manually zip the folder: {dist_dir}")
            else:
                print(f"\nWarning: Could not find executable at expected path: {exe_path}")
                print("Check the dist/ folder for build output.")

            return True
        else:
            print("\n" + "="*60)
            print("BUILD FAILED!")
            print("="*60)
            print("\nPyInstaller output:")
            print(result.stdout)
            if result.stderr:
                print("\nErrors:")
                print(result.stderr)
            return False

    except Exception as e:
        print(f"\nERROR: Failed to run PyInstaller: {e}")
        return False

def main():
    """Main build process."""
    print("="*60)
    print("PlethApp Windows Executable Builder")
    print("="*60)

    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Step 1: Clean previous builds
    clean_build_artifacts()

    # Step 2: Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Step 3: Check required files
    files_ok = check_required_files()
    if not files_ok:
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Build cancelled.")
            sys.exit(1)

    # Step 4: Build executable
    success = build_executable()

    if success:
        print("\nBuild process completed successfully!")
    else:
        print("\nBuild process failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == '__main__':
    main()