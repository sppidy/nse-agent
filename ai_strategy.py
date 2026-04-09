"""Gemini AI-powered trading strategy."""

import os
import json
import time
from google import genai
import pandas as pd
from dotenv import load_dotenv

import config
from strategy import add_indicators

load_dotenv()

# Rate limiting for free tier
_MIN_DELAY_BETWEEN_CALLS = 5  # seconds between API calls
_MAX_RETRIES = 3
# Fallback model chain — if one is rate-limited, try the next
# Model priority: Gemma 4 has 1500 RPD + unlimited TPM on free tier
# Gemini 2.5 Flash is smarter but only 20 RPD
_MODELS = [
    "gemma-4-31b-it",            # 1,500 RPD, unlimited TPM (primary)
    "gemini-2.5-flash",          # 20 RPD, 250K TPM (smarter, limited)
    "gemini-3.1-flash-lite-preview",  # 500 RPD, 250K TPM (fallback)
    "gemma-4-26b-a4b-it",        # 1,500 RPD, unlimited TPM (last resort)
]


def _get_client():
    """Initialize Gemini client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "Set your GEMINI_API_KEY in .env file.\n"
            "Get a free key at: https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=api_key)


def _call_gemini(client, prompt: str, retries: int = _MAX_RETRIES) -> str:
    """Call Gemini with model fallback and retry on rate limits."""
    for model in _MODELS:
        for attempt in range(retries):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                return response.text.strip()
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < retries - 1:
                        wait = _MIN_DELAY_BETWEEN_CALLS * (attempt + 2)
                        print(f"    Rate limited on {model}, waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"    {model} exhausted, trying next model...")
                        break  # try next model
                elif "404" in error_str or "NOT_FOUND" in error_str:
                    break  # model doesn't exist, try next
                else:
                    raise
    raise Exception("All Gemini models exhausted. Try again later or check your API quota at https://ai.dev/rate-limit")


def _prepare_stock_summary(symbol: str, df: pd.DataFrame) -> str:
    """Convert stock data into a text summary for Gemini."""
    df = add_indicators(df)
    if df.empty or len(df) < 5:
        return f"{symbol}: insufficient data"

    recent = df.tail(10)
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    price = latest["Close"]
    prev_close = prev["Close"]
    change_pct = ((price - prev_close) / prev_close) * 100
    high_5d = recent["High"].max()
    low_5d = recent["Low"].min()

    if len(df) >= 20:
        price_20d_ago = df.iloc[-20]["Close"]
        trend_20d = ((price - price_20d_ago) / price_20d_ago) * 100
    else:
        trend_20d = 0

    returns = df["Close"].pct_change().dropna()
    volatility = returns.tail(20).std() * 100

    avg_vol = df["Volume"].tail(20).mean()
    current_vol = latest["Volume"]
    vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1

    summary = (
        f"Symbol: {symbol}\n"
        f"Price: Rs.{price:.2f} ({change_pct:+.2f}%)\n"
        f"5D Range: Rs.{low_5d:.2f}-{high_5d:.2f}\n"
        f"20D Trend: {trend_20d:+.2f}%\n"
        f"RSI(14): {latest['rsi']:.1f}\n"
        f"EMA9: {latest['ema_short']:.2f}, EMA21: {latest['ema_long']:.2f} "
        f"({'Bullish' if latest['ema_short'] > latest['ema_long'] else 'Bearish'})\n"
        f"Volatility: {volatility:.2f}%, Vol: {vol_ratio:.1f}x avg\n"
    )

    summary += "Last 5 days: "
    for idx, row in recent.tail(5).iterrows():
        day_change = ((row["Close"] - row["Open"]) / row["Open"]) * 100
        summary += f"{row['Close']:.2f}({day_change:+.1f}%) "

    return summary


def analyze_batch(stock_data: list[tuple[str, pd.DataFrame]]) -> list[dict]:
    """Analyze ALL stocks in a single Gemini call to save API quota."""
    client = _get_client()

    summaries = []
    symbols = []
    for symbol, df in stock_data:
        summaries.append(_prepare_stock_summary(symbol, df))
        symbols.append(symbol)

    all_summaries = "\n---\n".join(summaries)

    # Get learning context from past trades
    from learner import get_learning_context
    learning = get_learning_context()

    # Get news sentiment
    from news_sentiment import get_sentiment_context
    news_context = get_sentiment_context(symbols)

    prompt = f"""You are a quantitative trading analyst that learns from past performance and reads market news.

{learning}

{news_context}

STOCKS DATA:
{all_summaries}

PORTFOLIO: Rs.{config.INITIAL_CAPITAL} capital, max {config.MAX_POSITION_SIZE_PCT*100}% per position, {config.STOP_LOSS_PCT*100}% stop loss, {config.TAKE_PROFIT_PCT*100}% take profit.

IMPORTANT RULES:
1. Use trade history to avoid repeating losing patterns and favor winning setups.
2. NEWS SENTIMENT is critical — if news is bearish for a stock, lower confidence even if technicals look good. If news is very bullish, boost confidence.
3. If overall market mood is BEARISH, be more cautious with all BUY signals.
4. If a company has negative news (scandal, downgrade, earnings miss), signal SELL or HOLD regardless of technicals.

Respond ONLY with a valid JSON array (no markdown, no extra text). One object per stock:
[
  {{
    "symbol": "SYMBOL.NS",
    "signal": "BUY" or "SELL" or "HOLD",
    "confidence": 0.0 to 1.0,
    "reason": "brief explanation",
    "entry_price": number or null,
    "stop_loss": number or null,
    "target": number or null
  }}
]"""

    try:
        text = _call_gemini(client, prompt)
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        results = json.loads(text)

        # Add price data back
        price_map = {}
        for symbol, df in stock_data:
            if not df.empty:
                price_map[symbol] = float(df["Close"].dropna().iloc[-1])
        for r in results:
            sym = r.get("symbol", "")
            r["price"] = price_map.get(sym, 0)

        return results
    except Exception as e:
        print(f"    Batch analysis failed: {e}")
        return [
            {
                "symbol": sym,
                "signal": "HOLD",
                "confidence": 0.0,
                "reason": f"AI analysis failed: {e}",
                "price": float(df["Close"].dropna().iloc[-1]) if not df.empty else 0,
            }
            for sym, df in stock_data
        ]


def analyze_single_stock(symbol: str, df: pd.DataFrame) -> dict:
    """Use Gemini to analyze a single stock."""
    client = _get_client()
    summary = _prepare_stock_summary(symbol, df)

    prompt = f"""You are a quantitative trading analyst. Analyze this Indian stock and provide a trading recommendation.

STOCK DATA:
{summary}

PORTFOLIO: Rs.{config.INITIAL_CAPITAL} capital, max {config.MAX_POSITION_SIZE_PCT*100}% per position, {config.STOP_LOSS_PCT*100}% stop loss, {config.TAKE_PROFIT_PCT*100}% take profit.

Respond ONLY with valid JSON (no markdown, no extra text):
{{
    "signal": "BUY" or "SELL" or "HOLD",
    "confidence": 0.0 to 1.0,
    "reason": "brief 1-2 sentence explanation",
    "entry_price": suggested entry price or null,
    "stop_loss": suggested stop loss price or null,
    "target": suggested target price or null
}}"""

    try:
        text = _call_gemini(client, prompt)
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        result = json.loads(text)
        result["symbol"] = symbol
        result["price"] = float(df["Close"].dropna().iloc[-1])
        return result
    except Exception as e:
        return {
            "symbol": symbol,
            "signal": "HOLD",
            "confidence": 0.0,
            "reason": f"AI analysis failed: {e}",
            "price": float(df["Close"].dropna().iloc[-1]) if not df.empty else 0,
        }


def analyze_watchlist(watchlist: list[str] | None = None) -> list[dict]:
    """Analyze all stocks in watchlist using a single batched Gemini call."""
    from data_fetcher import get_historical_data

    if watchlist is None:
        watchlist = config.WATCHLIST

    # Fetch all data first
    stock_data = []
    for symbol in watchlist:
        df = get_historical_data(symbol, period="60d", interval="1d")
        if not df.empty:
            stock_data.append((symbol, df))

    if not stock_data:
        return []

    # Single API call for all stocks (saves quota!)
    print(f"  Sending {len(stock_data)} stocks to Gemini in one batch...")
    return analyze_batch(stock_data)


def get_portfolio_advice(portfolio_summary: dict, signals: list[dict]) -> str:
    """Get Gemini's advice on overall portfolio strategy."""
    client = _get_client()

    prompt = f"""You are a trading advisor for a small Indian retail investor.

PORTFOLIO STATUS:
{json.dumps(portfolio_summary, indent=2)}

CURRENT SIGNALS:
{json.dumps(signals, indent=2)}

Give brief actionable advice (under 150 words):
1. Which trades to make now and why
2. Key risks
3. Strategy for Rs.{config.INITIAL_CAPITAL} capital"""

    try:
        text = _call_gemini(client, prompt)
        return text
    except Exception as e:
        return f"Could not get portfolio advice: {e}"
