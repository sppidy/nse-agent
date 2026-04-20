import unittest
import math
from unittest.mock import patch

import config
from paper_trader import PaperTrader, Portfolio, Position
from tests.helpers import workspace_temp_dir


class TestPaperTrader(unittest.TestCase):
    def test_buy_refreshes_portfolio_from_disk_before_save(self):
        with workspace_temp_dir() as tmp:
            path = f"{tmp}\\portfolio.json"
            Portfolio(cash=10000.0).save(path)
            trader = PaperTrader(filepath=path)

            with patch("paper_trader.is_market_trading_day", return_value=True), patch("paper_trader.random.uniform", return_value=1.0):
                trader.buy("AAA.NS", price=100.0, quantity=10)

            external = Portfolio.load(path)
            external.positions["MANUAL.NS"] = Position(
                symbol="MANUAL.NS",
                quantity=5,
                avg_price=50.0,
                entry_time="2026-01-01T00:00:00",
            )
            external.save(path)

            with patch("paper_trader.is_market_trading_day", return_value=True), patch("paper_trader.random.uniform", return_value=1.0):
                trader.buy("BBB.NS", price=100.0, quantity=10)

            final_data = Portfolio.load(path)
            self.assertIn("MANUAL.NS", final_data.positions)
            self.assertIn("AAA.NS", final_data.positions)
            self.assertIn("BBB.NS", final_data.positions)

    def test_buy_applies_ai_risk_targets_to_position(self):
        trader = PaperTrader(portfolio=Portfolio(cash=10000.0))
        ai_signal = {"stop_loss": 95.0, "target": 120.0}
        with patch("paper_trader.is_market_trading_day", return_value=True), patch("paper_trader.random.uniform", return_value=1.0):
            trader.buy("TEST.NS", price=100.0, quantity=10, confidence=0.8, ai_signal=ai_signal)

        pos = trader.portfolio.positions["TEST.NS"]
        self.assertAlmostEqual(95.0, pos.ai_stop_loss)
        self.assertAlmostEqual(120.0, pos.ai_target)
        self.assertGreater(pos.dynamic_take_profit_pct, config.TAKE_PROFIT_PCT)
        self.assertGreaterEqual(pos.dynamic_stop_loss_pct, config.MIN_STOP_LOSS_PCT)

    def test_buy_uses_capital_utilization_floor_for_low_confidence(self):
        original_initial = config.INITIAL_CAPITAL
        original_portfolios = dict(config.PORTFOLIOS)
        try:
            config.INITIAL_CAPITAL = 10000.0
            config.PORTFOLIOS["main"] = 10000.0
            trader = PaperTrader(portfolio=Portfolio(cash=10000.0))
            with patch("paper_trader.is_market_trading_day", return_value=True), patch("paper_trader.random.uniform", return_value=1.0):
                order = trader.buy("UTIL.NS", price=100.0, confidence=0.2, max_position_size_pct=0.10)
            self.assertIsNotNone(order)
            self.assertGreaterEqual(order.quantity, 4)
        finally:
            config.INITIAL_CAPITAL = original_initial
            config.PORTFOLIOS.clear()
            config.PORTFOLIOS.update(original_portfolios)

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
        with workspace_temp_dir() as tmp:
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

    def test_get_summary_ignores_nan_prices(self):
        portfolio = Portfolio(
            cash=1000.0,
            positions={"TEST.NS": Position(symbol="TEST.NS", quantity=2, avg_price=100.0, entry_time="2026-01-01T00:00:00")},
        )
        trader = PaperTrader(portfolio=portfolio)
        summary = trader.get_summary({"TEST.NS": float("nan")})
        self.assertTrue(math.isfinite(summary["total_value"]))
        self.assertEqual(1200.0, summary["total_value"])


if __name__ == "__main__":
    unittest.main()
