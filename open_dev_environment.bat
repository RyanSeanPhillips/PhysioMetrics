@echo off
REM Development Environment Launcher for PhysioMetrics
REM Opens terminal, Claude, VS Code, and File Explorer

set PROJECT_DIR=C:\Users\rphil2\Dropbox\python scripts\breath_analysis\pyqt6
set EXAMPLES_DIR=%PROJECT_DIR%\examples

REM Open Windows Terminal in project directory
start wt -d "%PROJECT_DIR%"

REM Open Claude (assuming default browser)
start https://claude.ai

REM Open VS Code in project directory
start code "%PROJECT_DIR%"

REM Open File Explorer in project directory
start explorer "%PROJECT_DIR%"

REM Open File Explorer in examples subdirectory
start explorer "%EXAMPLES_DIR%"

echo Development environment launched!
timeout /t 2
