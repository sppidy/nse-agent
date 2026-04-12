import tempfile
import unittest
from unittest.mock import patch

import config
from paper_trader import PaperTrader, Portfolio


class TestPaperTrader(unittest.TestCase):
    def test_buy_respects_explicit_max_position_size_pct(self):
        original_initial = config.INITIAL_CAPITAL
        try:
            config.INITIAL_CAPITAL = 1000.0
            trader = PaperTrader(portfolio=Portfolio(cash=1000.0))
            with patch("paper_trader.is_market_trading_day", return_value=True):
                order = trader.buy("TEST.NS", price=90.0, max_position_size_pct=0.1)
            self.assertIsNotNone(order)
            self.assertEqual(1, order.quantity)
        finally:
            config.INITIAL_CAPITAL = original_initial

    def test_portfolio_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            trader = PaperTrader(portfolio=Portfolio(cash=5000.0))
            with patch("paper_trader.is_market_trading_day", return_value=True):
                trader.buy("TEST.NS", price=100.0, quantity=5)
            path = f"{tmp}\\portfolio.json"
            trader.portfolio.save(path)
            loaded = Portfolio.load(path)
            self.assertIn("TEST.NS", loaded.positions)
            self.assertGreaterEqual(loaded.cash, 0)

    def test_buy_sell_blocked_on_non_trading_day(self):
        trader = PaperTrader(portfolio=Portfolio(cash=5000.0))
        with patch("paper_trader.is_market_trading_day", return_value=False):
            buy_order = trader.buy("TEST.NS", price=100.0, quantity=5)
            sell_order = trader.sell("TEST.NS", price=100.0, quantity=5)
            triggered = trader.check_stop_loss_take_profit({"TEST.NS": 95.0})
        self.assertIsNone(buy_order)
        self.assertIsNone(sell_order)
        self.assertEqual([], triggered)


if __name__ == "__main__":
    unittest.main()
