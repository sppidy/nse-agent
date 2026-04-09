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

    BUY when:
      - RSI < oversold threshold (stock is beaten down)
      - Short EMA crosses above Long EMA (momentum turning up)
      - Volume is above average (confirmation)

    SELL when:
      - RSI > overbought threshold (stock is overextended)
      - Short EMA crosses below Long EMA (momentum turning down)
    """
    df = add_indicators(df)
    df["signal"] = "HOLD"

    for i in range(1, len(df)):
        if pd.isna(df.iloc[i]["rsi"]) or pd.isna(df.iloc[i]["ema_long"]):
            continue

        rsi = df.iloc[i]["rsi"]
        ema_short = df.iloc[i]["ema_short"]
        ema_long = df.iloc[i]["ema_long"]
        prev_ema_short = df.iloc[i - 1]["ema_short"]
        prev_ema_long = df.iloc[i - 1]["ema_long"]
        volume = df.iloc[i]["Volume"]
        volume_sma = df.iloc[i]["volume_sma"]

        # BUY signal
        ema_crossover = prev_ema_short <= prev_ema_long and ema_short > ema_long
        rsi_oversold = rsi < config.RSI_OVERSOLD
        volume_confirm = pd.isna(volume_sma) or volume > volume_sma * 0.8

        if (ema_crossover or rsi_oversold) and volume_confirm:
            df.iloc[i, df.columns.get_loc("signal")] = "BUY"

        # SELL signal
        ema_crossunder = prev_ema_short >= prev_ema_long and ema_short < ema_long
        rsi_overbought = rsi > config.RSI_OVERBOUGHT

        if ema_crossunder or rsi_overbought:
            df.iloc[i, df.columns.get_loc("signal")] = "SELL"

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
