@echo off
setlocal
:: AI Trading Agent - Autopilot Launcher
:: Runs the trading bot during market hours (9:15 AM - 3:30 PM IST)

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Log output to file with date stamp
set "LOGFILE=autopilot_%date:~-4,4%%date:~-7,2%%date:~-10,2%.log"

echo [%date% %time%] Starting AI Trading Autopilot... >> %LOGFILE%
echo [%date% %time%] Starting AI Trading Autopilot...

set PYTHONUNBUFFERED=1
if not defined PYTHON_EXE set "PYTHON_EXE=python"
"%PYTHON_EXE%" -u autopilot.py --interval 15 >> "%LOGFILE%" 2>&1

echo [%date% %time%] Autopilot stopped. >> "%LOGFILE%"
