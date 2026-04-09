"""
Auto-updater for PhysioMetrics Windows builds.

Downloads release ZIPs from GitHub, extracts to staging, and generates
a batch script that swaps files after the app exits.

Pure Python — no Qt dependencies.
"""

import os
import sys
import shutil
import zipfile
import urllib.request
from pathlib import Path
from typing import Optional, Callable


def _get_appdata_dir() -> Path:
    """Get %APPDATA%/PhysioMetrics directory."""
    appdata = os.environ.get('APPDATA', os.path.expanduser('~'))
    return Path(appdata) / 'PhysioMetrics'


def _get_updates_dir() -> Path:
    """Get updates staging directory."""
    d = _get_appdata_dir() / 'updates'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_backup_dir() -> Path:
    """Get backup directory for rollback."""
    return _get_appdata_dir() / 'update_backup'


def is_running_as_bundle() -> bool:
    """Check if running as a PyInstaller frozen bundle."""
    return getattr(sys, 'frozen', False)


def get_install_dir() -> Optional[Path]:
    """Get the install directory when running as a frozen bundle."""
    if not is_running_as_bundle():
        return None
    return Path(sys.executable).parent


def download_update(
    url: str,
    filename: str,
    size: int,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """
    Download update ZIP to staging directory.

    Args:
        url: Download URL
        filename: Target filename
        size: Expected file size in bytes (0 if unknown)
        progress_callback: Called with (bytes_downloaded, total_bytes)

    Returns:
        Path to downloaded file

    Raises:
        OSError: On disk space or download failure
    """
    dest = _get_updates_dir() / filename

    # Check disk space (need ~2x size for download + extraction)
    needed = max(size * 3, 600_000_000)  # At least 600MB
    free = shutil.disk_usage(dest.parent).free
    if free < needed:
        needed_mb = needed // (1024 * 1024)
        free_mb = free // (1024 * 1024)
        raise OSError(
            f"Insufficient disk space: need ~{needed_mb} MB, "
            f"only {free_mb} MB available"
        )

    # Remove previous partial download
    if dest.exists():
        dest.unlink()

    req = urllib.request.Request(url)
    req.add_header('Accept', 'application/octet-stream')

    with urllib.request.urlopen(req, timeout=30) as response:
        total = int(response.headers.get('Content-Length', size))
        downloaded = 0
        chunk_size = 256 * 1024  # 256 KB

        with open(dest, 'wb') as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback:
                    progress_callback(downloaded, total)

    return dest


def verify_download(zip_path: Path) -> bool:
    """Verify ZIP integrity. Returns True if valid."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            bad = zf.testzip()
            return bad is None
    except (zipfile.BadZipFile, OSError):
        return False


def extract_update(zip_path: Path) -> Path:
    """
    Extract ZIP to staging directory.

    Returns:
        Path to extracted directory containing PhysioMetrics.exe
    """
    staging_base = _get_updates_dir()

    # Clean any previous extraction
    for item in staging_base.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(staging_base)

    # Find the extracted directory (ZIP contains a top-level folder)
    extracted_dirs = [
        d for d in staging_base.iterdir()
        if d.is_dir() and d.name.startswith('PhysioMetrics')
    ]
    if not extracted_dirs:
        raise FileNotFoundError("No PhysioMetrics directory found in ZIP")

    staging_dir = extracted_dirs[0]

    # Verify the exe exists
    exe_path = staging_dir / 'PhysioMetrics.exe'
    if not exe_path.exists():
        raise FileNotFoundError(
            f"PhysioMetrics.exe not found in extracted directory: {staging_dir}"
        )

    return staging_dir


def generate_update_script(install_dir: Path, staging_dir: Path, pid: int) -> Path:
    """
    Generate apply_update.cmd that swaps files after app exits.

    Args:
        install_dir: Current installation directory
        staging_dir: Directory with new version files
        pid: Current process PID to wait for

    Returns:
        Path to generated .cmd script
    """
    script_path = _get_updates_dir() / 'apply_update.cmd'
    backup_dir = _get_backup_dir()

    # Use raw strings for Windows paths in batch
    install = str(install_dir)
    staging = str(staging_dir)
    backup = str(backup_dir)
    exe = str(install_dir / 'PhysioMetrics.exe')

    script = f'''@echo off
setlocal enabledelayedexpansion
title PhysioMetrics Update
echo.
echo ============================================
echo   PhysioMetrics Auto-Update
echo ============================================
echo.
echo Waiting for PhysioMetrics to close...

:: Wait for app to exit (poll every 2s, timeout 60s)
set /a TIMEOUT=0
:wait_loop
tasklist /FI "PID eq {pid}" 2>NUL | find /I "{pid}" >NUL
if %ERRORLEVEL% NEQ 0 goto :app_closed
timeout /t 2 /nobreak >NUL
set /a TIMEOUT+=2
if %TIMEOUT% GEQ 60 (
    echo.
    echo ERROR: PhysioMetrics did not close within 60 seconds.
    echo Please close the application manually and try again.
    pause
    goto :cleanup_fail
)
goto :wait_loop

:app_closed
echo PhysioMetrics has closed.
echo.

:: Small delay for file handles to release
timeout /t 1 /nobreak >NUL

:: Backup current installation
echo Backing up current version...
if exist "{backup}" (
    rmdir /s /q "{backup}" 2>NUL
)
mkdir "{backup}" 2>NUL
robocopy "{install}" "{backup}" /E /NFL /NDL /NJH /NJS /NC /NS >NUL 2>&1
if not exist "{backup}\\PhysioMetrics.exe" (
    echo WARNING: Backup may be incomplete, but continuing...
)
echo Backup complete.
echo.

:: Copy new files
echo Installing update...
robocopy "{staging}" "{install}" /E /NFL /NDL /NJH /NJS /NC /NS >NUL 2>&1

:: Verify new exe exists
if not exist "{exe}" (
    echo.
    echo ERROR: Update failed - PhysioMetrics.exe not found after copy!
    echo Restoring backup...
    robocopy "{backup}" "{install}" /E /NFL /NDL /NJH /NJS /NC /NS >NUL 2>&1
    echo Backup restored. Your previous version should work.
    pause
    goto :cleanup_fail
)

echo Update installed successfully!
echo.

:: Launch new version
echo Starting PhysioMetrics...
start "" "{exe}"

:: Clean up staging (not backup — keep for rollback)
echo Cleaning up...
rmdir /s /q "{staging}" 2>NUL

:: Self-delete
echo.
echo Update complete!
timeout /t 2 /nobreak >NUL
del "%~f0" 2>NUL
exit /b 0

:cleanup_fail
:: Don't delete staging on failure so user can retry
exit /b 1
'''

    script_path.write_text(script, encoding='utf-8')
    return script_path


def generate_rollback_script(install_dir: Path, backup_dir: Path, pid: int) -> Path:
    """
    Generate rollback script that restores the backup version.

    Args:
        install_dir: Current installation directory
        backup_dir: Backup directory with previous version
        pid: Current process PID to wait for

    Returns:
        Path to generated .cmd script
    """
    script_path = _get_updates_dir() / 'apply_rollback.cmd'

    install = str(install_dir)
    backup = str(backup_dir)
    exe = str(install_dir / 'PhysioMetrics.exe')

    script = f'''@echo off
setlocal enabledelayedexpansion
title PhysioMetrics Rollback
echo.
echo ============================================
echo   PhysioMetrics Version Rollback
echo ============================================
echo.
echo Waiting for PhysioMetrics to close...

:: Wait for app to exit
set /a TIMEOUT=0
:wait_loop
tasklist /FI "PID eq {pid}" 2>NUL | find /I "{pid}" >NUL
if %ERRORLEVEL% NEQ 0 goto :app_closed
timeout /t 2 /nobreak >NUL
set /a TIMEOUT+=2
if %TIMEOUT% GEQ 60 (
    echo ERROR: PhysioMetrics did not close within 60 seconds.
    pause
    exit /b 1
)
goto :wait_loop

:app_closed
echo PhysioMetrics has closed.
timeout /t 1 /nobreak >NUL
echo.

:: Move current version to temp
set "TEMP_DIR=%TEMP%\\physiometrics_rollback_temp"
if exist "%TEMP_DIR%" rmdir /s /q "%TEMP_DIR%" 2>NUL
echo Moving current version aside...
robocopy "{install}" "%TEMP_DIR%" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS >NUL 2>&1

:: Restore backup
echo Restoring previous version...
robocopy "{backup}" "{install}" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS >NUL 2>&1

:: Verify exe exists
if not exist "{exe}" (
    echo ERROR: Rollback failed!
    echo Restoring current version...
    robocopy "%TEMP_DIR%" "{install}" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS >NUL 2>&1
    pause
    exit /b 1
)

echo Rollback complete!
echo.

:: Clean up temp
rmdir /s /q "%TEMP_DIR%" 2>NUL

:: Launch restored version
echo Starting PhysioMetrics...
start "" "{exe}"

echo.
timeout /t 2 /nobreak >NUL
del "%~f0" 2>NUL
exit /b 0
'''

    script_path.write_text(script, encoding='utf-8')
    return script_path


def apply_update(zip_path: Path) -> Path:
    """
    Orchestrate: verify → extract → generate update script.

    Args:
        zip_path: Path to downloaded ZIP

    Returns:
        Path to the generated update script

    Raises:
        ValueError: If ZIP is corrupt
        FileNotFoundError: If exe not found in bundle or ZIP
    """
    install_dir = get_install_dir()
    if install_dir is None:
        raise RuntimeError("Not running as a frozen bundle")

    if not verify_download(zip_path):
        zip_path.unlink(missing_ok=True)
        raise ValueError("Downloaded file is corrupt. Please try again.")

    staging_dir = extract_update(zip_path)
    script = generate_update_script(install_dir, staging_dir, os.getpid())
    return script


def has_backup() -> bool:
    """Check if a backup exists for rollback."""
    backup_dir = _get_backup_dir()
    return (backup_dir / 'PhysioMetrics.exe').exists()


def get_backup_version() -> Optional[str]:
    """Try to read version from backup's version_info module."""
    backup_dir = _get_backup_dir()
    version_file = backup_dir / 'version_info.py'
    if not version_file.exists():
        # Try looking in _internal
        version_file = backup_dir / '_internal' / 'version_info.py'
    if not version_file.exists():
        return None

    try:
        text = version_file.read_text(encoding='utf-8')
        for line in text.splitlines():
            if line.startswith('VERSION_STRING'):
                # Parse: VERSION_STRING = "1.0.15-beta.3"
                return line.split('=', 1)[1].strip().strip('"\'')
    except Exception:
        pass
    return None


def clean_updates() -> None:
    """Remove staging files (not backup)."""
    updates_dir = _get_updates_dir()
    for item in updates_dir.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            elif item.suffix in ('.zip', '.cmd'):
                item.unlink()
        except OSError:
            pass
