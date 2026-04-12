# Setup Windows Task Scheduler for AI Trading Autopilot
# Run this script as Administrator: Right-click PowerShell -> Run as Administrator
# Then: powershell -ExecutionPolicy Bypass -File B:\projects\ai-trading-agent\setup_scheduler.ps1

$TaskName = "AI-Trading-Autopilot"
$Description = "Runs AI Trading Agent autopilot during Indian market hours (Mon-Fri)"
$WorkingDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BatchFile = Join-Path $WorkingDir "start_autopilot.bat"

# Remove existing task if present
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

# Trigger: Monday-Friday at 9:00 AM IST (before market opens at 9:15)
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "09:00AM"

# Action: Run the batch file
$Action = New-ScheduledTaskAction -Execute $BatchFile -WorkingDirectory $WorkingDir

# Settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 8) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5)

# Register the task (runs as current user)
Register-ScheduledTask `
    -TaskName $TaskName `
    -Description $Description `
    -Trigger $Trigger `
    -Action $Action `
    -Settings $Settings `
    -RunLevel Limited

Write-Host ""
Write-Host "Task '$TaskName' registered successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Schedule: Monday-Friday at 9:00 AM"
Write-Host "Action:   $BatchFile"
Write-Host "Log:      $WorkingDir\autopilot_YYYYMMDD.log"
Write-Host ""
Write-Host "To check:  Get-ScheduledTask -TaskName '$TaskName'"
Write-Host "To run now: Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "To remove:  Unregister-ScheduledTask -TaskName '$TaskName'"
Write-Host ""
