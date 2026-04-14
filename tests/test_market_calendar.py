import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd

import market_calendar


class TestMarketCalendar(unittest.TestCase):
    def setUp(self):
        market_calendar._day_cache.clear()

    @patch("market_calendar.yf.Ticker")
    def test_today_stale_intraday_is_closed_during_market_hours(self, ticker_cls):
        stale_idx = pd.date_range(
            "2026-04-13 09:15:00",
            periods=5,
            freq="1min",
            tz=market_calendar.IST,
        )
        stale_df = pd.DataFrame({"Close": [1, 2, 3, 4, 5]}, index=stale_idx)

        ticker = ticker_cls.return_value
        ticker.history.return_value = stale_df

        with patch(
            "market_calendar.now_ist",
            return_value=datetime(2026, 4, 14, 12, 0, tzinfo=market_calendar.IST),
        ):
            self.assertFalse(market_calendar.is_market_trading_day(datetime(2026, 4, 14).date()))

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

        with patch(
            "market_calendar.now_ist",
            return_value=datetime(2026, 4, 14, 8, 30, tzinfo=market_calendar.IST),
        ):
            self.assertTrue(market_calendar.is_market_trading_day(day))

        with patch(
            "market_calendar.now_ist",
            return_value=datetime(2026, 4, 14, 12, 0, tzinfo=market_calendar.IST),
        ):
            self.assertFalse(market_calendar.is_market_trading_day(day))


if __name__ == "__main__":
    unittest.main()
