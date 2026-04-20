import unittest
from datetime import datetime

import market_calendar


class TestMarketCalendar(unittest.TestCase):
    def setUp(self):
        market_calendar._day_cache.clear()

    def test_weekend_always_closed(self):
        """Weekends should always return False."""
        # Saturday 2026-04-11
        self.assertFalse(market_calendar.is_market_trading_day(datetime(2026, 4, 11).date()))
        # Sunday 2026-04-12
        self.assertFalse(market_calendar.is_market_trading_day(datetime(2026, 4, 12).date()))

    def test_known_nse_holiday_closed(self):
        """Republic Day is a fixed NSE holiday."""
        # 2026-01-26 is a Monday Republic Day -> closed
        self.assertFalse(market_calendar.is_market_trading_day(datetime(2026, 1, 26).date()))

    def test_known_trading_day_open(self):
        """A regular Monday with no holiday should be open."""
        # 2026-04-20 is a regular Monday (already verified via mcal.schedule on deploy)
        self.assertTrue(market_calendar.is_market_trading_day(datetime(2026, 4, 20).date()))

    def test_cache_populates(self):
        """Repeat calls should hit the cache rather than re-querying."""
        day = datetime(2026, 4, 20).date()
        self.assertTrue(market_calendar.is_market_trading_day(day))
        self.assertIn(day, market_calendar._day_cache)


if __name__ == "__main__":
    unittest.main()
