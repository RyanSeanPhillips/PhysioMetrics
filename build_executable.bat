@echo off
REM ====================================================================
REM PlethApp Build Script - Creates Windows Executable
REM ====================================================================
REM This script automates the process of building a Windows executable
REM from the PlethApp breath analysis application using PyInstaller.
REM ====================================================================

echo.
echo ====================================================================
echo Building PlethApp Windows Executable
echo ====================================================================
echo.

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo ERROR: PyInstaller is not installed!
    echo Please install it with: pip install pyinstaller
    echo.
    pause
    exit /b 1
)

REM Clean previous builds (keep dist/ for version history)
echo Cleaning build artifacts...
if exist "build" rmdir /s /q "build"
if exist "__pycache__" rmdir /s /q "__pycache__"
if exist "dist" (
    echo Keeping dist\ folder to preserve version history
) else (
    echo Creating dist\ folder...
    mkdir "dist"
)

REM Clean Python cache files
for /r %%i in (*.pyc) do del "%%i" 2>nul
for /r %%i in (__pycache__) do rmdir /s /q "%%i" 2>nul

echo.
echo Starting PyInstaller build...
echo This may take several minutes...
echo.

REM Get version string from version_info.py
for /f "tokens=*" %%i in ('python -c "from version_info import VERSION_STRING; print(VERSION_STRING)"') do set VERSION=%%i

REM Build the executable using the spec file
pyinstaller --clean pleth_app.spec

REM Check if build was successful (looking for versioned output)
if exist "dist\PlethApp_v%VERSION%\PlethApp_v%VERSION%.exe" (
    echo.
    echo ====================================================================
    echo BUILD SUCCESSFUL!
    echo ====================================================================
    echo.
    echo Executable created: dist\PlethApp_v%VERSION%\PlethApp_v%VERSION%.exe
    echo Build directory: dist\PlethApp_v%VERSION%\
    echo File size:
    for %%A in ("dist\PlethApp_v%VERSION%\PlethApp_v%VERSION%.exe") do echo %%~zA bytes
    echo.
    echo You can now distribute the entire folder to users.
    echo The executable is self-contained and doesn't require Python installation.
    echo.
    echo Previous versions are preserved in the dist\ folder.
    echo.

    REM Create zip file for distribution
    echo Creating zip file for distribution...
    python -c "import shutil; shutil.make_archive('dist/PlethApp_v%VERSION%_Windows', 'zip', 'dist', 'PlethApp_v%VERSION%')"
    if exist "dist\PlethApp_v%VERSION%_Windows.zip" (
        echo.
        echo ZIP FILE CREATED!
        for %%A in ("dist\PlethApp_v%VERSION%_Windows.zip") do echo Zip size: %%~zA bytes (%%~zA / 1048576 MB^)
        echo Location: dist\PlethApp_v%VERSION%_Windows.zip
        echo.
        echo Ready to upload to GitHub releases!
    ) else (
        echo.
        echo Warning: Failed to create zip file
        echo You can manually zip the folder: dist\PlethApp_v%VERSION%
    )
    echo.
) else (
    echo.
    echo ====================================================================
    echo BUILD FAILED!
    echo ====================================================================
    echo.
    echo Please check the output above for error messages.
    echo Common issues:
    echo - Missing dependencies in requirements.txt
    echo - UI files not found in ui\ directory
    echo - Icon files not found in images\ directory
    echo.
)

echo.
echo Build process completed.
pause