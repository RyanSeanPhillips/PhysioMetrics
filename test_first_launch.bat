@echo off
REM Test PlethApp telemetry by simulating first-time launch
REM This deletes the config folder and runs the app

echo ================================================
echo Testing PlethApp First Launch
echo ================================================
echo.

REM Delete config folder to simulate first-time user
if exist "%APPDATA%\PlethApp" (
    echo Deleting existing config: %APPDATA%\PlethApp
    rmdir /s /q "%APPDATA%\PlethApp"
    echo Config deleted.
) else (
    echo No existing config found.
)

echo.
echo Starting PlethApp...
echo ================================================
echo.

REM Run the app
python run_debug.py

echo.
echo ================================================
echo App closed.
echo ================================================
pause
