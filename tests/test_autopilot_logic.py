import unittest

import pandas as pd

from autopilot import _adjust_confidence, _sized_position_pct, _trend_score


def _make_hist(days: int = 30, uptrend: bool = True) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=days, freq="D")
    if uptrend:
        close = [100 + i * 1.2 for i in range(days)]
    else:
        close = [100 - i * 0.8 for i in range(days)]
    return pd.DataFrame(
        {
            "Close": close,
            "Volume": [100000 + (i * 500) for i in range(days)],
        },
        index=idx,
    )


class TestAutopilotLogic(unittest.TestCase):
    def test_trend_score_prefers_uptrend(self):
        bullish = _trend_score(_make_hist(uptrend=True))
        bearish = _trend_score(_make_hist(uptrend=False))
        self.assertIsNotNone(bullish)
        self.assertIsNotNone(bearish)
        self.assertGreater(bullish, bearish)

    def test_adjust_confidence_penalizes_ml_disagreement(self):
        base = 0.8
        adjusted, ml_agrees = _adjust_confidence(
            base, "BUY", {"prediction": "DOWN", "confidence": 0.8}, True, "NEUTRAL"
        )
        self.assertFalse(ml_agrees)
        self.assertLess(adjusted, base)

    def test_sized_position_pct_caps_in_bear(self):
        sized = _sized_position_pct(ai_size_pct=0.9, adjusted_conf=0.95, regime="BEAR")
        self.assertLessEqual(sized, 0.3)
        self.assertGreaterEqual(sized, 0.01)


if __name__ == "__main__":
    unittest.main()
