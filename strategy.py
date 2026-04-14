"""Trading strategies using technical indicators."""

import pandas as pd
import ta

import config


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI and EMA indicators to OHLCV dataframe."""
    df = df.copy()
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=config.RSI_PERIOD).rsi()
    df["ema_short"] = ta.trend.EMAIndicator(df["Close"], window=config.EMA_SHORT).ema_indicator()
    df["ema_long"] = ta.trend.EMAIndicator(df["Close"], window=config.EMA_LONG).ema_indicator()
    df["volume_sma"] = df["Volume"].rolling(window=20).mean()
    
    # MACD
    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df["Close"])
    df["bb_high"] = bollinger.bollinger_hband()
    df["bb_low"] = bollinger.bollinger_lband()
    df["bb_mid"] = bollinger.bollinger_mavg()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]
    
    # Average True Range (ATR)
    df["atr"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    
    return df


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate BUY/SELL signals based on RSI + EMA crossover strategy.
    """
    df = add_indicators(df)
    df["signal"] = "HOLD"

    prev_ema_short = df["ema_short"].shift(1)
    prev_ema_long = df["ema_long"].shift(1)

    ema_crossover = (prev_ema_short <= prev_ema_long) & (df["ema_short"] > df["ema_long"])
    rsi_oversold = df["rsi"] < config.RSI_OVERSOLD
    volume_confirm = df["volume_sma"].isna() | (df["Volume"] > df["volume_sma"] * 0.8)

    buy_cond = (ema_crossover | rsi_oversold) & volume_confirm

    ema_crossunder = (prev_ema_short >= prev_ema_long) & (df["ema_short"] < df["ema_long"])
    rsi_overbought = df["rsi"] > config.RSI_OVERBOUGHT

    sell_cond = ema_crossunder | rsi_overbought

    df.loc[buy_cond, "signal"] = "BUY"
    df.loc[sell_cond, "signal"] = "SELL"

    return df


def get_latest_signal(symbol: str, df: pd.DataFrame) -> dict:
    """Get the latest signal for a symbol."""
    df = generate_signals(df)
    if df.empty:
        return {"symbol": symbol, "signal": "HOLD", "reason": "No data"}

    last = df.iloc[-1]
    reason_parts = []
    if not pd.isna(last.get("rsi")):
        reason_parts.append(f"RSI={last['rsi']:.1f}")
    if not pd.isna(last.get("ema_short")) and not pd.isna(last.get("ema_long")):
        if last["ema_short"] > last["ema_long"]:
            reason_parts.append("EMA bullish")
        else:
            reason_parts.append("EMA bearish")

    return {
        "symbol": symbol,
        "signal": last["signal"],
        "price": round(last["Close"], 2),
        "rsi": round(last["rsi"], 1) if not pd.isna(last.get("rsi")) else None,
        "reason": ", ".join(reason_parts),
    }
