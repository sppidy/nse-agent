"""Market calendar helpers for NSE trading-day checks."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import yfinance as yf

import config
from logger import logger

IST = timezone(timedelta(hours=5, minutes=30))
MARKET_OPEN_TIME = time(hour=9, minute=15)
MARKET_CLOSE_TIME = time(hour=15, minute=30)

_day_cache: dict[date, bool] = {}


def now_ist() -> datetime:
    """Return current timestamp in IST."""
    return datetime.now(IST)


def is_market_trading_day(day: date | None = None) -> bool:
    """
    Return True if NSE has a trading session on the given day.

    For future dates we conservatively assume weekday sessions.
    For today/past dates we validate against index data.
    """
    if day is None:
        day = now_ist().date()

    if day.weekday() > 4:
        return False

    today = now_ist().date()
    if day > today:
        return True

    if day in _day_cache:
        return _day_cache[day]

    try:
        ticker = yf.Ticker(config.MARKET_INDEX)

        if day == today:
            intraday = ticker.history(period="1d", interval="1m")
            if not intraday.empty:
                _day_cache[day] = True
                return True

        next_day = day + timedelta(days=1)
        daily = ticker.history(
            start=day.strftime("%Y-%m-%d"),
            end=next_day.strftime("%Y-%m-%d"),
            interval="1d",
        )
        is_open = not daily.empty
        _day_cache[day] = is_open
        return is_open
    except Exception as exc:
        logger.warning(f"Could not verify NSE trading day for {day}: {exc}. Defaulting to closed.")
        _day_cache[day] = False
        return False
