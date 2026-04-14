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
            intraday = ticker.history(period="5d", interval="1m")
            if not intraday.empty:
                idx = intraday.index
                try:
                    idx_ist = idx.tz_convert(IST) if getattr(idx, "tz", None) else idx.tz_localize(IST)
                except Exception:
                    idx_ist = idx
                latest_bar_day = idx_ist.max().date()
                if latest_bar_day == today:
                    _day_cache[day] = True
                    return True

            # Before opening bell, today's bars may not exist yet.
            # Do not cache this assumption so we can re-evaluate once market hours start.
            if now_ist().time() < MARKET_OPEN_TIME:
                return True
            _day_cache[day] = False
            return False

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
