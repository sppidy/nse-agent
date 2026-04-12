import unittest

import pandas as pd

from backtester import backtest_portfolio


def _make_df(seed: int) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=80, freq="D")
    base = 100 + seed
    close = [base + i * 0.5 + ((-1) ** i) * 0.8 for i in range(80)]
    return pd.DataFrame(
        {
            "Open": [c - 0.2 for c in close],
            "High": [c + 1.0 for c in close],
            "Low": [c - 1.0 for c in close],
            "Close": close,
            "Volume": [100000 + i * 100 for i in range(80)],
        },
        index=idx,
    )


class TestBacktestPortfolio(unittest.TestCase):
    def test_shared_capital_backtest_returns_metrics(self):
        data = {
            "AAA.NS": _make_df(0),
            "BBB.NS": _make_df(5),
        }
        result = backtest_portfolio(data, initial_capital=100000.0)
        self.assertNotIn("error", result)
        self.assertIn("final_equity", result)
        self.assertIn("total_return_pct", result)
        self.assertGreater(result["final_equity"], 0)


if __name__ == "__main__":
    unittest.main()
