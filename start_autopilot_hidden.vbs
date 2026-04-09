' AI Trading Agent - Headless Autopilot Launcher
' Runs the batch file without showing a console window

Set WshShell = CreateObject("WScript.Shell")
WshShell.Run """B:\projects\ai-trading-agent\start_autopilot.bat""", 0, False
