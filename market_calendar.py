"""Market calendar helpers for NSE trading-day checks.

Uses pandas_market_calendars (XNSE) as the authoritative source — holidays,
half-days and weekends are resolved deterministically without any API call.
No yfinance heuristics, no fallback needed.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import pandas as pd
import pandas_market_calendars as mcal

from logger import logger

IST = timezone(timedelta(hours=5, minutes=30))
MARKET_OPEN_TIME = time(hour=9, minute=15)
MARKET_CLOSE_TIME = time(hour=15, minute=30)

_NSE = mcal.get_calendar("XNSE")
_day_cache: dict[date, bool] = {}


def now_ist() -> datetime:
    """Return current timestamp in IST."""
    return datetime.now(IST)


def is_market_trading_day(day: date | None = None) -> bool:
    """Return True if NSE has a trading session on the given day."""
    if day is None:
        day = now_ist().date()

    if day in _day_cache:
        return _day_cache[day]

    if day.weekday() > 4:
        _day_cache[day] = False
        return False

    try:
        day_str = day.strftime("%Y-%m-%d")
        schedule = _NSE.schedule(start_date=day_str, end_date=day_str)
        is_open = not schedule.empty
    except Exception as exc:
        # pandas_market_calendars is deterministic — failures are structural
        # (library issue, not a data outage). Weekday fallback keeps the agent
        # alive rather than blocking all trading.
        logger.warning(f"NSE calendar lookup failed for {day}: {exc}. Defaulting to weekday-open.")
        is_open = day.weekday() <= 4

    _day_cache[day] = is_open
    return is_open
