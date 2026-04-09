#!/bin/bash
# AI Trading Agent - Autopilot Launcher (Linux/Server)

cd "$(dirname "$0")"

LOGFILE="autopilot_$(date +%Y%m%d).log"

echo "[$(date)] Starting AI Trading Autopilot..." | tee -a "$LOGFILE"

export PYTHONUNBUFFERED=1
python3 -u autopilot.py --interval 15 2>&1 | tee -a "$LOGFILE"

echo "[$(date)] Autopilot stopped." | tee -a "$LOGFILE"
