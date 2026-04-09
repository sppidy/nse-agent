@echo off
:: AI Trading Agent - Autopilot Launcher
:: Runs the trading bot during market hours (9:15 AM - 3:30 PM IST)

cd /d B:\projects\ai-trading-agent

:: Log output to file with date stamp
set LOGFILE=autopilot_%date:~-4,4%%date:~-7,2%%date:~-10,2%.log

echo [%date% %time%] Starting AI Trading Autopilot... >> %LOGFILE%
echo [%date% %time%] Starting AI Trading Autopilot...

set PYTHONUNBUFFERED=1
C:\Python314\python.exe -u autopilot.py --interval 15 >> %LOGFILE% 2>&1

echo [%date% %time%] Autopilot stopped. >> %LOGFILE%
