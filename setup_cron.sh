#!/bin/bash
# Alternative to systemd: setup cron job for autopilot
# Runs Mon-Fri at 9:00 AM IST, stops at 3:45 PM IST

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$WORK_DIR/venv/bin/python3"

# Add cron entries
(crontab -l 2>/dev/null; cat <<CRON
# AI Trading Agent - Start autopilot Mon-Fri 9:00 AM IST
0 9 * * 1-5 cd $WORK_DIR && PYTHONUNBUFFERED=1 $PYTHON -u autopilot.py --interval 15 >> $WORK_DIR/logs/autopilot_\$(date +\%Y\%m\%d).log 2>&1 &
# AI Trading Agent - Kill autopilot Mon-Fri 3:45 PM IST
45 15 * * 1-5 pkill -f "autopilot.py --interval" 2>/dev/null
CRON
) | crontab -

echo "Cron jobs installed:"
echo "  START: Mon-Fri 9:00 AM"
echo "  STOP:  Mon-Fri 3:45 PM"
echo ""
echo "Verify with: crontab -l"
echo "NOTE: Ensure your server timezone is set to IST (Asia/Kolkata)"
echo "  sudo timedatectl set-timezone Asia/Kolkata"
