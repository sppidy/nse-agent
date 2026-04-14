"""Configuration for the paper trading agent."""

import json
import os
import time

# Project root directory (so files are found regardless of cwd)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Starting capital in INR
INITIAL_CAPITAL = 10_000_000.0  # Rs. 1 crore for paper trading data harvesting
MARKET_INDEX = "^NSEI"  # NIFTY 50 index for market regime detection

# Default stocks to track (NSE tickers for yfinance use .NS suffix)
# Expanded watchlist: original 10 + trending NSE stocks (affordable, under Rs.1200)
WATCHLIST = [
    # ── Original 10 ──
    "ITC.NS",
    "SBIN.NS",
    "TATASTEEL.NS",
    "WIPRO.NS",
    "PNB.NS",
    "NHPC.NS",
    "IRFC.NS",
    "SAIL.NS",
    "IOC.NS",
    "IDEA.NS",
    # ── Trending additions ──
    "SUZLON.NS",       # Green energy, strong momentum
    "BPCL.NS",         # Oil & gas, high volume
    "GAIL.NS",         # Gas utility, steady gainer
    "BHEL.NS",         # Capital goods, infra play
    "YESBANK.NS",      # Banking turnaround, penny stock
    "HFCL.NS",         # Telecom infra, affordable
    "BANKBARODA.NS",   # PSU bank, strong rally
    "NBCC.NS",         # Construction, govt contracts
    "SJVN.NS",         # Hydro power, green energy
    "ADANIPOWER.NS",   # Power sector, trending
]

# Trading parameters
MAX_POSITION_SIZE_PCT = 0.10  # Max 10% of capital per stock (Rs.600)
MAX_OPEN_POSITIONS = 20  # No restriction — diversify across all watchlist stocks
BROKERAGE_PER_ORDER = 0.0  # Paper trading = no fees (set to 20 for live)
SLIPPAGE_PCT = 0.001  # 0.1% simulated slippage

# Strategy parameters (RSI + EMA)
RSI_PERIOD = 14
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65
EMA_SHORT = 9
EMA_LONG = 21
STOP_LOSS_PCT = 0.02  # 2% stop loss
TAKE_PROFIT_PCT = 0.03  # 3% take profit
DYNAMIC_TRAILING_ENABLED = True
MIN_STOP_LOSS_PCT = 0.008
MAX_STOP_LOSS_PCT = 0.05
MIN_TAKE_PROFIT_PCT = 0.015
MAX_TAKE_PROFIT_PCT = 0.12
TRAILING_CONFIDENCE_SCALE = 0.6
TRAILING_PROFIT_LOCK_SCALE = 0.35

# Capital utilization controls
CAPITAL_DEPLOYMENT_TARGET_PCT = 0.9
CAPITAL_UTILIZATION_MIN_BET_PCT = 0.02

# Intraday settings
INTRADAY_LEVERAGE = 5  # 5x margin for intraday (simulated)
USE_LEVERAGE = False  # Enable when comfortable with the strategy

# Data settings
DATA_INTERVAL = "5m"  # 5-minute candles for intraday
BACKTEST_DAYS = 60  # Days of historical data for backtesting

# ── Hot-reload from config_overrides.json ─────────────────────
# Drop a config_overrides.json in PROJECT_DIR to change settings at runtime.
# Only the keys listed in _RELOADABLE are honoured; unknown keys are ignored.
# The file is re-read at most once every 60 seconds.

_OVERRIDES_FILE = os.path.join(PROJECT_DIR, "config_overrides.json")
_RELOADABLE = {
    "MAX_POSITION_SIZE_PCT", "MAX_OPEN_POSITIONS",
    "STOP_LOSS_PCT", "TAKE_PROFIT_PCT", "DYNAMIC_TRAILING_ENABLED",
    "MIN_STOP_LOSS_PCT", "MAX_STOP_LOSS_PCT", "MIN_TAKE_PROFIT_PCT",
    "MAX_TAKE_PROFIT_PCT", "RSI_OVERSOLD", "RSI_OVERBOUGHT",
    "CAPITAL_DEPLOYMENT_TARGET_PCT", "CAPITAL_UTILIZATION_MIN_BET_PCT",
    "DATA_INTERVAL", "BACKTEST_DAYS",
}
_last_reload: float = 0
_RELOAD_INTERVAL = 60  # seconds


def reload_overrides(force: bool = False) -> list[str]:
    """Re-read config_overrides.json and apply any reloadable settings.

    Returns list of keys that were updated (empty if nothing changed).
    Called automatically by autopilot at the start of each cycle.
    """
    global _last_reload
    now = time.time()
    if not force and (now - _last_reload) < _RELOAD_INTERVAL:
        return []
    _last_reload = now

    if not os.path.exists(_OVERRIDES_FILE):
        return []

    try:
        with open(_OVERRIDES_FILE, "r") as f:
            overrides = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    changed = []
    mod = globals()
    for key, value in overrides.items():
        if key in _RELOADABLE and mod.get(key) != value:
            mod[key] = value
            changed.append(key)
    return changed
