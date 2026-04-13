import unittest
from unittest.mock import Mock, patch

import pandas as pd

import data_fetcher


class TestDataFetcher(unittest.TestCase):
    def test_clean_price_filters_invalid_values(self):
        self.assertIsNone(data_fetcher._clean_price(None))
        self.assertIsNone(data_fetcher._clean_price(float("nan")))
        self.assertIsNone(data_fetcher._clean_price(float("inf")))
        self.assertIsNone(data_fetcher._clean_price(0))
        self.assertEqual(101.25, data_fetcher._clean_price(101.25))

    def test_get_live_price_falls_back_to_history(self):
        ticker = Mock()
        ticker.fast_info = {"lastPrice": float("nan")}
        ticker.history.return_value = pd.DataFrame({"Close": [102.5]})
        with patch("data_fetcher.yf.Ticker", return_value=ticker):
            self.assertEqual(102.5, data_fetcher.get_live_price("TEST.NS"))

    def test_get_live_price_returns_none_for_invalid_sources(self):
        ticker = Mock()
        ticker.fast_info = {"lastPrice": float("nan")}
        ticker.history.return_value = pd.DataFrame({"Close": [float("nan")]})
        with patch("data_fetcher.yf.Ticker", return_value=ticker):
            self.assertIsNone(data_fetcher.get_live_price("TEST.NS"))


if __name__ == "__main__":
    unittest.main()
