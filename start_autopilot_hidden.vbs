' AI Trading Agent - Headless Autopilot Launcher
' Runs the batch file without showing a console window

Set WshShell = CreateObject("WScript.Shell")
scriptDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
WshShell.Run """" & scriptDir & "\start_autopilot.bat""", 0, False
