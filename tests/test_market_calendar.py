import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd

import market_calendar


class TestMarketCalendar(unittest.TestCase):
    def setUp(self):
        market_calendar._day_cache.clear()

    @patch("market_calendar.yf.Ticker")
    def test_today_stale_intraday_defaults_open_during_market_hours(self, ticker_cls):
        """During market hours, stale yfinance data should NOT mark the day as closed."""
        stale_idx = pd.date_range(
            "2026-04-13 09:15:00",
            periods=5,
            freq="1min",
            tz=market_calendar.IST,
        )
        stale_df = pd.DataFrame({"Close": [1, 2, 3, 4, 5]}, index=stale_idx)

        ticker = ticker_cls.return_value
        ticker.history.return_value = stale_df

        # At noon with stale data => still defaults to open (not cached)
        with patch(
            "market_calendar.now_ist",
            return_value=datetime(2026, 4, 14, 12, 0, tzinfo=market_calendar.IST),
        ):
            self.assertTrue(market_calendar.is_market_trading_day(datetime(2026, 4, 14).date()))
            self.assertNotIn(datetime(2026, 4, 14).date(), market_calendar._day_cache)

    @patch("market_calendar.yf.Ticker")
    def test_today_stale_intraday_closed_after_market_hours(self, ticker_cls):
        """After market close, stale yfinance data means it was a holiday."""
        stale_idx = pd.date_range(
            "2026-04-13 09:15:00",
            periods=5,
            freq="1min",
            tz=market_calendar.IST,
        )
        stale_df = pd.DataFrame({"Close": [1, 2, 3, 4, 5]}, index=stale_idx)

        ticker = ticker_cls.return_value
        ticker.history.return_value = stale_df

        # At 4 PM with stale data => holiday, cached as False
        with patch(
            "market_calendar.now_ist",
            return_value=datetime(2026, 4, 14, 16, 0, tzinfo=market_calendar.IST),
        ):
            self.assertFalse(market_calendar.is_market_trading_day(datetime(2026, 4, 14).date()))
            self.assertFalse(market_calendar._day_cache[datetime(2026, 4, 14).date()])

    @patch("market_calendar.yf.Ticker")
    def test_preopen_assumption_not_cached_for_today(self, ticker_cls):
        stale_idx = pd.date_range(
            "2026-04-13 09:15:00",
            periods=5,
            freq="1min",
            tz=market_calendar.IST,
        )
        stale_df = pd.DataFrame({"Close": [1, 2, 3, 4, 5]}, index=stale_idx)
        ticker = ticker_cls.return_value
        ticker.history.return_value = stale_df
        day = datetime(2026, 4, 14).date()

        # Pre-open: should return True and NOT cache
        with patch(
            "market_calendar.now_ist",
            return_value=datetime(2026, 4, 14, 8, 30, tzinfo=market_calendar.IST),
        ):
            self.assertTrue(market_calendar.is_market_trading_day(day))
            self.assertNotIn(day, market_calendar._day_cache)

        # During market hours with stale data: should still return True
        with patch(
            "market_calendar.now_ist",
            return_value=datetime(2026, 4, 14, 12, 0, tzinfo=market_calendar.IST),
        ):
            self.assertTrue(market_calendar.is_market_trading_day(day))

    @patch("market_calendar.yf.Ticker")
    def test_exception_defaults_open_during_market_hours(self, ticker_cls):
        """Network errors during market hours should default to open."""
        ticker_cls.side_effect = Exception("Network error")

        with patch(
            "market_calendar.now_ist",
            return_value=datetime(2026, 4, 14, 10, 0, tzinfo=market_calendar.IST),
        ):
            self.assertTrue(market_calendar.is_market_trading_day(datetime(2026, 4, 14).date()))
            self.assertNotIn(datetime(2026, 4, 14).date(), market_calendar._day_cache)

    def test_weekend_always_closed(self):
        """Weekends should always return False."""
        # Saturday
        self.assertFalse(market_calendar.is_market_trading_day(datetime(2026, 4, 11).date()))
        # Sunday
        self.assertFalse(market_calendar.is_market_trading_day(datetime(2026, 4, 12).date()))


if __name__ == "__main__":
    unittest.main()
