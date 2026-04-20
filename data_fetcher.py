"""Fetch market data for Indian stocks — Groww primary, yfinance fallback."""

import math
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import config
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from logger import logger

try:
    import groww_client
except ImportError:  # keep data_fetcher importable if the module goes missing
    groww_client = None


def _clean_price(value) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price) or price <= 0:
        return None
    return price

def _try_groww_candles(symbol: str, period: str, interval: str, label: str) -> pd.DataFrame | None:
    """Return Groww-sourced OHLCV, or None if Groww isn't configured / failed."""
    if not (groww_client and groww_client.is_configured()):
        return None
    try:
        df = groww_client.fetch_candles(symbol, period=period, interval=interval)
        if df.empty:
            logger.warning(f"Groww returned empty {label} for {symbol} — falling back to yfinance")
            return None
        return df
    except Exception as e:
        logger.warning(f"Groww {label} failed for {symbol}: {e} — falling back to yfinance")
        return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logger.warning(f"Retrying yfinance API call. Attempt {retry_state.attempt_number}")
)
def _yf_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        logger.warning(f"Warning: No data returned for {symbol} ({period}/{interval})")
        return df
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    return df


def get_historical_data(symbol: str, period: str = "60d", interval: str = "1d") -> pd.DataFrame:
    """Fetch historical OHLCV — Groww first, then yfinance on failure."""
    df = _try_groww_candles(symbol, period, interval, "historical")
    if df is not None:
        return df
    try:
        return _yf_history(symbol, period, interval)
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        raise e


def get_intraday_data(symbol: str, days: int = 5, interval: str = "5m") -> pd.DataFrame:
    """Fetch intraday OHLCV — Groww first, then yfinance on failure."""
    period = f"{days}d"
    df = _try_groww_candles(symbol, period, interval, "intraday")
    if df is not None:
        return df
    try:
        return _yf_history(symbol, period, interval)
    except Exception as e:
        logger.error(f"Error fetching intraday data for {symbol}: {e}")
        raise e


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logger.warning(f"Retrying yfinance API call. Attempt {retry_state.attempt_number}")
)
def _yf_live_price(symbol: str) -> float | None:
    ticker = yf.Ticker(symbol)
    try:
        info = ticker.fast_info
        if info is not None:
            last = getattr(info, 'lastPrice', None) or (info.get("lastPrice") if hasattr(info, 'get') else None)
            live = _clean_price(last)
            if live is not None:
                return live
    except Exception:
        pass
    hist = ticker.history(period="1d")
    if not hist.empty:
        return _clean_price(hist["Close"].iloc[-1])
    return None


def get_live_price(symbol: str) -> float | None:
    """Get the latest available price for a symbol — Groww LTP first, yfinance fallback."""
    if groww_client and groww_client.is_configured():
        try:
            price = groww_client.fetch_live_price(symbol)
            if price is not None:
                return price
        except groww_client.GrowwUnsupported:
            # Expected skip (e.g., indices). Silent fall-through.
            pass
        except Exception as e:
            logger.warning(f"Groww LTP failed for {symbol}: {e} — falling back to yfinance")
    try:
        return _yf_live_price(symbol)
    except Exception as e:
        logger.error(f"Error fetching live price for {symbol}: {e}")
        raise e


def get_watchlist_prices(watchlist: list[str] | None = None) -> dict[str, float]:
    """Get current prices for all watchlist stocks."""
    if watchlist is None:
        watchlist = config.WATCHLIST
    prices = {}
    for symbol in watchlist:
        price = get_live_price(symbol)
        if price is not None:
            prices[symbol] = price
    return prices


def get_market_regime() -> str:
    """
    Detect the broader market regime using the NIFTY 50 index.
    Returns: 'BULL', 'BEAR', or 'NEUTRAL'
    """
    try:
        df = get_historical_data(config.MARKET_INDEX, period="1y", interval="1d")
        if df.empty or len(df) < 200:
            return "NEUTRAL"
            
        current_price = df["Close"].iloc[-1]
        sma_50 = df["Close"].rolling(window=50).mean().iloc[-1]
        sma_200 = df["Close"].rolling(window=200).mean().iloc[-1]
        
        if current_price > sma_50 and sma_50 > sma_200:
            return "BULL"
        elif current_price < sma_200:
            return "BEAR"
        else:
            return "NEUTRAL"
    except Exception as e:
        logger.error(f"Error detecting market regime: {e}")
        return "NEUTRAL"
def get_stock_info(symbol: str) -> dict:
    """Get basic stock info."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        return {
            "symbol": symbol,
            "price": info.get("lastPrice", 0),
            "market_cap": info.get("marketCap", 0),
            "day_high": info.get("dayHigh", 0),
            "day_low": info.get("dayLow", 0),
        }
    except Exception as e:
        logger.error(f"Error fetching info for {symbol}: {e}")
        return {"symbol": symbol}
