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


def get_scored_signal(symbol: str, df: pd.DataFrame) -> dict:
    """Multi-indicator scored signal for rule-based trading.

    Scores each indicator independently and combines them into a confidence
    value. Requires agreement from multiple indicators before emitting a
    BUY or SELL — much more selective than the simple crossover signal.

    Returns dict with: symbol, signal, confidence, price, reason, indicators.
    """
    df = add_indicators(df)
    if df.empty or len(df) < 5:
        return {"symbol": symbol, "signal": "HOLD", "confidence": 0, "reason": "Insufficient data"}

    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = last["Close"]
    reasons = []
    buy_score = 0
    sell_score = 0

    # ── 1. RSI (weight: 2) ──
    rsi = last.get("rsi")
    if pd.notna(rsi):
        if rsi < config.RSI_OVERSOLD:
            buy_score += 2
            reasons.append(f"RSI oversold ({rsi:.0f})")
        elif rsi < 45:
            buy_score += 1
            reasons.append(f"RSI low ({rsi:.0f})")
        elif rsi > 80:
            sell_score += 2
            reasons.append(f"RSI extreme ({rsi:.0f})")
        elif rsi > config.RSI_OVERBOUGHT:
            sell_score += 1
            reasons.append(f"RSI overbought ({rsi:.0f})")

    # ── 2. EMA crossover (weight: 2) ──
    ema_s, ema_l = last.get("ema_short"), last.get("ema_long")
    prev_ema_s, prev_ema_l = prev.get("ema_short"), prev.get("ema_long")
    if pd.notna(ema_s) and pd.notna(ema_l) and pd.notna(prev_ema_s) and pd.notna(prev_ema_l):
        if prev_ema_s <= prev_ema_l and ema_s > ema_l:
            buy_score += 2
            reasons.append("EMA bullish crossover")
        elif ema_s > ema_l:
            buy_score += 1
        if prev_ema_s >= prev_ema_l and ema_s < ema_l:
            sell_score += 2
            reasons.append("EMA bearish crossover")
        elif ema_s < ema_l:
            sell_score += 1

    # ── 3. MACD (weight: 2) ──
    macd_diff = last.get("macd_diff")
    prev_macd_diff = prev.get("macd_diff")
    if pd.notna(macd_diff) and pd.notna(prev_macd_diff):
        if prev_macd_diff <= 0 < macd_diff:
            buy_score += 2
            reasons.append("MACD bullish cross")
        elif macd_diff > 0:
            buy_score += 1
        if prev_macd_diff >= 0 > macd_diff:
            sell_score += 2
            reasons.append("MACD bearish cross")
        elif macd_diff < 0:
            sell_score += 1

    # ── 4. Bollinger Bands (weight: 1) ──
    bb_low = last.get("bb_low")
    bb_high = last.get("bb_high")
    if pd.notna(bb_low) and pd.notna(bb_high):
        if price <= bb_low:
            buy_score += 1
            reasons.append("Price at lower BB")
        elif price >= bb_high:
            sell_score += 1
            reasons.append("Price at upper BB")

    # ── 5. Volume confirmation (weight: 1) ──
    vol_sma = last.get("volume_sma")
    if pd.notna(vol_sma) and vol_sma > 0:
        vol_ratio = last["Volume"] / vol_sma
        if vol_ratio > 1.5:
            # Volume confirms whichever direction is winning
            if buy_score > sell_score:
                buy_score += 1
                reasons.append(f"Vol {vol_ratio:.1f}x avg")
            elif sell_score > buy_score:
                sell_score += 1
                reasons.append(f"Vol {vol_ratio:.1f}x avg")

    # ── 6. Trend consistency — 5-day close direction (weight: 1) ──
    if len(df) >= 5:
        closes_5d = df["Close"].tail(5)
        up_days = (closes_5d.diff().dropna() > 0).sum()
        if up_days >= 4:
            buy_score += 1
            reasons.append(f"{up_days}/4 up days")
        elif up_days <= 1:
            sell_score += 1
            reasons.append(f"{4 - up_days}/4 down days")

    # ── Combine scores ──
    max_possible = 9  # 2+2+2+1+1+1
    # Require 4+ score with clear directional edge (score gap >= 2)
    if buy_score >= 4 and buy_score >= sell_score + 2:
        confidence = round(min(buy_score / max_possible + 0.2, 0.85), 2)
        signal = "BUY"
    elif sell_score >= 4 and sell_score >= buy_score + 2:
        confidence = round(min(sell_score / max_possible + 0.2, 0.85), 2)
        signal = "SELL"
    else:
        confidence = 0
        signal = "HOLD"

    return {
        "symbol": symbol,
        "signal": signal,
        "confidence": confidence,
        "price": round(float(price), 2),
        "reason": ", ".join(reasons) if reasons else "No strong signals",
        "buy_score": buy_score,
        "sell_score": sell_score,
        "indicators": {
            "rsi": round(float(rsi), 1) if pd.notna(rsi) else None,
            "ema_trend": "bullish" if (pd.notna(ema_s) and pd.notna(ema_l) and ema_s > ema_l) else "bearish",
            "macd_diff": round(float(macd_diff), 3) if pd.notna(macd_diff) else None,
        },
    }
