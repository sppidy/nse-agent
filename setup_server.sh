#!/bin/bash
# Janus - Server Setup Script
# Run once after cloning the repo on your server

set -e

echo "=== Janus - Server Setup ==="

# 1. Create virtual environment
echo "[1/5] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
echo "[2/5] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Setup .env
if [ ! -f .env ]; then
    echo "[3/5] Creating .env from template..."
    cp .env.example .env
    echo "  >> Edit .env and add your GEMINI_API_KEY"
    echo "  >> Get a free key at: https://aistudio.google.com/apikey"
else
    echo "[3/5] .env already exists, skipping..."
fi

# 4. Create required directories
echo "[4/5] Creating directories..."
mkdir -p logs models

# 5. Install systemd service
echo "[5/5] Installing systemd service..."
WORK_DIR="$(pwd)"
PYTHON_PATH="$(pwd)/venv/bin/python3"

cat > /tmp/ai-trading-agent.service <<SVCEOF
[Unit]
Description=Janus Autopilot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$WORK_DIR
ExecStart=$PYTHON_PATH -u autopilot.py --interval 15
Restart=on-failure
RestartSec=60
Environment=PYTHONUNBUFFERED=1

# Logging
StandardOutput=append:$WORK_DIR/logs/autopilot.log
StandardError=append:$WORK_DIR/logs/autopilot.log

[Install]
WantedBy=multi-user.target
SVCEOF

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your GEMINI_API_KEY"
echo "     nano .env"
echo ""
echo "  2. Test manually first:"
echo "     source venv/bin/activate"
echo "     python main.py status"
echo "     python main.py autopilot --force  # test outside market hours"
echo ""
echo "  3. Install the systemd service (runs Mon-Fri during market hours):"
echo "     sudo cp /tmp/ai-trading-agent.service /etc/systemd/system/"
echo "     sudo systemctl daemon-reload"
echo "     sudo systemctl enable ai-trading-agent"
echo "     sudo systemctl start ai-trading-agent"
echo ""
echo "  4. Check logs:"
echo "     sudo journalctl -u ai-trading-agent -f"
echo "     tail -f logs/autopilot.log"
echo ""
