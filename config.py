"""Configuration for the paper trading agent."""

import json
import os
import time

# Project root directory (so files are found regardless of cwd)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Parallel paper portfolios — same signals, different starting capital.
# 'main' = large capital for ML/learner data harvesting; 'eval' = realistic small
# capital so you can evaluate how the system behaves with tight position sizing.
PORTFOLIOS: dict[str, float] = {
    "main": 10_000_000.0,  # Rs. 1 crore — data harvesting
    "eval": 10_000.0,      # Rs. 10,000 — realistic evaluation
}
DEFAULT_PORTFOLIO = "main"

# Per-portfolio position-sizing overrides. A Rs.10k portfolio cannot use the
# same 10% cap as Rs.1cr — 10% of 10k = Rs.1000 which prices out most stocks
# (1 share ≈ Rs.500-1000). Widen eval's cap and shrink its max open positions
# so the basket is concentrated enough to actually deploy capital.
PORTFOLIO_MAX_POSITION_PCT: dict[str, float] = {
    "main": 0.10,
    "eval": 0.25,
}
PORTFOLIO_MAX_OPEN_POSITIONS: dict[str, int] = {
    "main": 20,
    "eval": 4,  # 4 × 25% = 100% — fully deployable
}

# Per-portfolio confidence thresholds by regime. `eval` runs aggressive on
# purpose — the whole point is to evaluate whether more permissive entry
# rules generate alpha vs. the conservative `main` portfolio. Bear-regime
# threshold is the strictest; bull is loosest.
PORTFOLIO_CONFIDENCE_THRESHOLD: dict[str, dict[str, float]] = {
    "main": {"BULL": 0.72, "NEUTRAL": 0.72, "BEAR": 0.82},
    "eval": {"BULL": 0.55, "NEUTRAL": 0.60, "BEAR": 0.68},
}

# Force the CatBoost ML model to influence trades regardless of the 75%
# maturity gate. `eval` opts in so we can observe how the live (still
# immature) model actually performs — it gets the full ±penalty/boost
# treatment in confidence adjustment. `main` stays gated for safety.
PORTFOLIO_ML_OVERRIDE: dict[str, bool] = {
    "main": False,
    "eval": True,
}

# Backward-compat scalar — many modules still reference INITIAL_CAPITAL directly.
INITIAL_CAPITAL = PORTFOLIOS[DEFAULT_PORTFOLIO]
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
