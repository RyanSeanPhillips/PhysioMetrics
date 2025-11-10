# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for PhysioMetrics - Breath Analysis Application
This file configures how PyInstaller packages the application into a Windows executable.
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Get the directory containing this spec file
spec_root = os.path.dirname(os.path.abspath(SPEC))

# Import version for naming the executable
import sys
sys.path.insert(0, spec_root)
from version_info import VERSION_STRING

block_cipher = None

# Collect additional data files
added_files = [
    # UI files
    (os.path.join(spec_root, 'ui', '*.ui'), 'ui'),

    # Application icons and assets
    (os.path.join(spec_root, 'images'), 'images'),
    (os.path.join(spec_root, 'assets'), 'assets'),

    # Core modules
    (os.path.join(spec_root, 'core'), 'core'),
]

# Hidden imports - modules that PyInstaller might miss
hidden_imports = [
    # PyQt6 modules
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.uic',

    # Scientific computing
    'numpy',
    'pandas',
    'scipy',
    'scipy.signal',
    'scipy.ndimage',

    # Matplotlib backends
    'matplotlib',
    'matplotlib.backends.backend_qtagg',
    'matplotlib.backends.backend_qt5agg',
    'matplotlib.figure',
    'matplotlib.pyplot',

    # Core application modules
    'core.state',
    'core.abf_io',
    'core.filters',
    'core.plotting',
    'core.stim',
    'core.peaks',
    'core.metrics',

    # Standard library modules that might be missed
    'csv',
    'json',
    'pathlib',
    're',
    'sys',
    'os',

    # Fix for setuptools issues (excluding pkg_resources entirely)
    'setuptools',

    # Add missing modules that SciPy/NumPy need
    'unittest',
    'unittest.mock',
    'numpy.testing',
    'scipy.signal._support_alternative_backends',
    'scipy._lib._array_api',
    'scipy._lib.array_api_compat',
]

a = Analysis(
    ['main.py'],
    pathex=[spec_root],
    binaries=[],
    datas=added_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unused modules to reduce size
        'tkinter',
        'test',
        # Exclude PySide6 to avoid Qt binding conflicts
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        # Exclude problematic modules
        'pkg_resources',
        # Exclude numba and dependencies (not used in production, only in test files)
        'numba',
        'llvmlite',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Single file executable (SLOW startup - commented out)
# exe = EXE(
#     pyz,
#     a.scripts,
#     a.binaries,
#     a.zipfiles,
#     a.datas,
#     [],
#     name='PhysioMetrics',
#     debug=False,
#     bootloader_ignore_signals=False,
#     strip=False,
#     upx=True,
#     upx_exclude=[],
#     runtime_tmpdir=None,
#     console=False,  # Set to True for debugging
#     disable_windowed_traceback=False,
#     target_arch=None,
#     codesign_identity=None,
#     entitlements_file=None,
#     icon=os.path.join(spec_root, 'images', 'plethapp_thumbnail_dark_round.ico'),
#     version_file=None,  # Can add version info later
# )

# Directory distribution (FAST startup)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=f'PhysioMetrics_v{VERSION_STRING}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(spec_root, 'images', 'plethapp_thumbnail_dark_round.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=f'PhysioMetrics_v{VERSION_STRING}',
)