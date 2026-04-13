"""Configuration for the paper trading agent."""

import os

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
