"""AI-powered trading strategy using Groq (primary), Copilot/Haiku, and Gemini (fallback)."""

import os
import json
import time
import re
import asyncio
import concurrent.futures
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import config
from logger import logger
from strategy import add_indicators

load_dotenv()

# Rate limiting
_MIN_DELAY_BETWEEN_CALLS = 2  # seconds between API calls (Groq is fast)
_MAX_RETRIES = 3

# Copilot proxy (GitHub Copilot -> OpenAI-compatible API on localhost)
_COPILOT_PROXY_URL = os.getenv("COPILOT_PROXY_URL", "http://localhost:4141/v1")
_COPILOT_MODELS = [
    "claude-haiku-4.5",    # Fast, smart, free via Copilot
    "gpt-4o-mini",         # Cheap fallback
]

# Groq models (primary) — ordered by capability, diversified to spread rate limits
# Each model has its own TPM/RPM quota on free tier, so more models = more headroom
_GROQ_MODELS = [
    "llama-3.3-70b-versatile",      # Best quality, 6K TPM free
    "qwen/qwen3-32b",               # Good quality, 6K TPM free
    "meta-llama/llama-4-scout-17b-16e-instruct",  # Fast, 6K TPM free
    "mistral-saba-24b",             # Mistral quality, 6K TPM free
    "llama-3.1-8b-instant",          # Fastest fallback, 14.4K RPD
]

# Gemini models (fallback only)
_GEMINI_MODELS = [
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

class SignalSchema(BaseModel):
    symbol: str
    signal: str = Field(description="'BUY', 'SELL', or 'HOLD'")
    confidence: float = Field(description="0.0 to 1.0")
    position_size_pct: float = Field(description="0.01 to 1.0")
    reason: str
    entry_price: float | None = None
    stop_loss: float | None = None
    target: float | None = None

def _sanitize_prompt_text(text: str, limit: int = 240) -> str:
    cleaned = re.sub(r"https?://\S+", "", text or "")
    cleaned = cleaned.replace("```", "")
    cleaned = re.sub(r"(?i)(ignore\s+previous\s+instructions|system\s+prompt|developer\s+message)", "[filtered]", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:limit]


def _normalize_signal_record(raw: dict | SignalSchema, symbol: str, fallback_price: float) -> dict:
    if isinstance(raw, SignalSchema):
        raw = raw.model_dump()
    
    signal = str(raw.get("signal", "HOLD")).upper()
    if signal not in {"BUY", "SELL", "HOLD"}:
        signal = "HOLD"

    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))
    try:
        position_size_pct = float(raw.get("position_size_pct", confidence))
    except (TypeError, ValueError):
        position_size_pct = confidence
    position_size_pct = max(0.01, min(position_size_pct, 1.0))

    def _safe_num(v):
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return {
        "symbol": symbol,
        "signal": signal,
        "confidence": confidence,
        "reason": _sanitize_prompt_text(str(raw.get("reason", "No explanation provided")), limit=280),
        "entry_price": _safe_num(raw.get("entry_price")),
        "stop_loss": _safe_num(raw.get("stop_loss")),
        "target": _safe_num(raw.get("target")),
        "position_size_pct": position_size_pct,
        "price": float(fallback_price),
    }


def _get_groq_client():
    """Initialize Groq client."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    from groq import AsyncGroq
    return AsyncGroq(api_key=api_key)


def _get_gemini_client():
    """Initialize Gemini client (fallback)."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        return None
    from google import genai
    return genai.Client(api_key=api_key)


# Keep a reference for backward compatibility (chat.py, api_server.py)
def _get_client():
    """Return Groq client if available, else Gemini."""
    client = _get_groq_client()
    if client:
        return client
    client = _get_gemini_client()
    if client:
        return client
    raise ValueError("Set GROQ_API_KEY or GEMINI_API_KEY in .env file.")


def _clean_json_text(raw_text: str) -> str:
    """Strip markdown fences and trailing garbage from JSON text."""
    clean = raw_text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*\n?", "", clean)
        clean = re.sub(r"\n?```\s*$", "", clean)
        clean = clean.strip()
    # Sometimes models append extra data after the JSON array
    if clean.startswith("["):
        last_bracket = clean.rfind("]")
        if last_bracket > 0:
            clean = clean[:last_bracket + 1]
    return clean


def _parse_signals_from_text(raw_text: str) -> list[SignalSchema] | None:
    """Try to parse SignalSchema list from raw JSON text."""
    clean = _clean_json_text(raw_text)
    if not clean:
        return None
    try:
        raw_json = json.loads(clean)
        # Groq json_object mode may wrap array in {"signals": [...]} or {"results": [...]}
        if isinstance(raw_json, dict):
            for key in ("signals", "results", "data", "stocks", "analysis"):
                if key in raw_json and isinstance(raw_json[key], list):
                    raw_json = raw_json[key]
                    break
            else:
                # Single signal object — wrap in list
                if "symbol" in raw_json:
                    raw_json = [raw_json]
                else:
                    return None
        if isinstance(raw_json, list):
            results = []
            for item in raw_json:
                try:
                    results.append(SignalSchema(**(item if isinstance(item, dict) else {})))
                except Exception:
                    pass
            if results:
                return results
    except json.JSONDecodeError as je:
        logger.warning(f"    JSON parse failed at char {je.pos}: {je.msg}")
    return None


async def _call_groq_async(prompt: str, retries: int = _MAX_RETRIES, want_json: bool = False) -> str | list[SignalSchema]:
    """Call Groq API with model fallback and retry on rate limits."""
    client = _get_groq_client()
    if not client:
        raise Exception("GROQ_API_KEY not set")

    kwargs = {}
    if want_json:
        kwargs["response_format"] = {"type": "json_object"}

    for model in _GROQ_MODELS:
        for attempt in range(retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=4096,
                    **kwargs,
                )
                text = response.choices[0].message.content.strip()

                if want_json:
                    # Try to parse signals
                    signals = _parse_signals_from_text(text)
                    if signals:
                        logger.info(f"    Groq/{model}: parsed {len(signals)} signals")
                        return signals
                    # Model returned JSON but not an array — try wrapping
                    logger.warning(f"    Groq/{model}: JSON response didn't yield signals, trying next model")
                    break  # try next model
                return text
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate_limit" in error_str.lower():
                    if attempt < retries - 1:
                        wait = _MIN_DELAY_BETWEEN_CALLS * (attempt + 2)
                        logger.warning(f"    Groq/{model} rate limited, waiting {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        logger.warning(f"    Groq/{model} exhausted retries, trying next model...")
                        break
                elif "404" in error_str or "not_found" in error_str.lower() or "decommissioned" in error_str.lower():
                    logger.warning(f"    Groq/{model} unavailable/decommissioned, trying next model...")
                    break
                else:
                    logger.warning(f"    Groq/{model} error: {e}")
                    if attempt < retries - 1:
                        await asyncio.sleep(_MIN_DELAY_BETWEEN_CALLS)
                    else:
                        break
    raise Exception("All Groq models exhausted")


async def _call_copilot_async(prompt: str, retries: int = _MAX_RETRIES, want_json: bool = False) -> str | list[SignalSchema]:
    """Call Claude Haiku via Copilot proxy (OpenAI-compatible)."""
    import httpx

    for model in _COPILOT_MODELS:
        for attempt in range(retries):
            try:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 4096,
                }
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(
                        f"{_COPILOT_PROXY_URL}/chat/completions",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    resp.raise_for_status()
                    data = resp.json()

                text = data["choices"][0]["message"]["content"].strip()

                if want_json:
                    signals = _parse_signals_from_text(text)
                    if signals:
                        logger.info(f"    Copilot/{model}: parsed {len(signals)} signals")
                        return signals
                    logger.warning(f"    Copilot/{model}: didn't yield signals, trying next model")
                    break
                return text
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate" in error_str.lower():
                    if attempt < retries - 1:
                        wait = _MIN_DELAY_BETWEEN_CALLS * (attempt + 2)
                        logger.warning(f"    Copilot/{model} rate limited, waiting {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        logger.warning(f"    Copilot/{model} exhausted retries, trying next model...")
                        break
                elif "connect" in error_str.lower() or "refused" in error_str.lower():
                    logger.warning(f"    Copilot proxy unavailable: {e}")
                    raise Exception("Copilot proxy not running")
                else:
                    logger.warning(f"    Copilot/{model} error: {e}")
                    if attempt < retries - 1:
                        await asyncio.sleep(_MIN_DELAY_BETWEEN_CALLS)
                    else:
                        break
    raise Exception("All Copilot models exhausted")


async def _call_gemini_async(client, prompt: str, retries: int = _MAX_RETRIES, response_schema=None) -> str | list[SignalSchema]:
    """Call Gemini with model fallback and retry on rate limits (fallback)."""
    from google.genai import types

    kwargs = {}
    if response_schema:
        kwargs["config"] = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
        )

    for model in _GEMINI_MODELS:
        for attempt in range(retries):
            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    **kwargs
                )
                if response_schema:
                    parsed = response.parsed
                    if parsed is not None:
                        return parsed
                    raw_text = (response.text or "").strip()
                    logger.warning(f"    Gemini/{model}: response.parsed is None, trying manual parse")
                    signals = _parse_signals_from_text(raw_text)
                    if signals:
                        logger.info(f"    Gemini/{model}: manual parse recovered {len(signals)} signals")
                        return signals
                    continue
                return response.text.strip()
            except Exception as e:
                error_str = str(e)
                if (
                    "429" in error_str
                    or "RESOURCE_EXHAUSTED" in error_str
                    or "503" in error_str
                    or "UNAVAILABLE" in error_str
                ):
                    if attempt < retries - 1:
                        wait = _MIN_DELAY_BETWEEN_CALLS * (attempt + 2)
                        logger.warning(f"    Gemini/{model} temporarily unavailable, waiting {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        logger.warning(f"    Gemini/{model} exhausted, trying next model...")
                        break
                elif "404" in error_str or "NOT_FOUND" in error_str:
                    break
                elif "schema" in error_str.lower() and response_schema:
                    logger.warning(f"    Gemini/{model} schema error: {e}")
                    break
                else:
                    raise
    raise Exception("All Gemini models exhausted.")


async def _call_ai_async(prompt: str, want_json: bool = False) -> str | list[SignalSchema]:
    """Call Groq first, then Copilot/Haiku, then Gemini as final fallback."""
    # Try Groq first (fast, free)
    if os.getenv("GROQ_API_KEY"):
        try:
            return await _call_groq_async(prompt, want_json=want_json)
        except Exception as e:
            logger.warning(f"    Groq failed, trying Copilot: {e}")

    # Try Copilot proxy (Claude Haiku via GitHub Copilot)
    try:
        return await _call_copilot_async(prompt, want_json=want_json)
    except Exception as e:
        logger.warning(f"    Copilot failed, falling back to Gemini: {e}")

    # Final fallback: Gemini
    gemini_client = _get_gemini_client()
    if gemini_client:
        schema = list[SignalSchema] if want_json else None
        return await _call_gemini_async(gemini_client, prompt, response_schema=schema)

    raise Exception("No AI provider available. Set GROQ_API_KEY or GEMINI_API_KEY.")


def _call_gemini(client, prompt: str, retries: int = _MAX_RETRIES) -> str:
    """Sync compatibility wrapper — uses Groq first, Gemini as fallback."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_call_ai_async(prompt))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(
            lambda: asyncio.run(_call_ai_async(prompt))
        ).result()


def _prepare_stock_summary(symbol: str, df: pd.DataFrame) -> str:
    """Convert stock data into a text summary for AI."""
    df = add_indicators(df)
    if df.empty or len(df) < 5:
        return f"{symbol}: insufficient data"

    recent = df.tail(10)
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    price = latest["Close"]
    prev_close = prev["Close"]
    change_pct = ((price - prev_close) / prev_close) * 100 if prev_close != 0 else 0
    high_5d = recent["High"].max()
    low_5d = recent["Low"].min()

    if len(df) >= 20:
        price_20d_ago = df.iloc[-20]["Close"]
        trend_20d = ((price - price_20d_ago) / price_20d_ago) * 100 if price_20d_ago != 0 else 0
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
        day_change = ((row["Close"] - row["Open"]) / row["Open"]) * 100 if row["Open"] != 0 else 0
        summary += f"{row['Close']:.2f}({day_change:+.1f}%) "

    return summary


async def analyze_batch_async(stock_data: list[tuple[str, pd.DataFrame]]) -> list[dict]:
    """Analyze ALL stocks in a single AI call (Groq primary, Gemini fallback)."""

    summaries = []
    symbols = []
    for symbol, df in stock_data:
        summaries.append(_prepare_stock_summary(symbol, df))
        symbols.append(symbol)
    price_map = {
        symbol: float(df["Close"].dropna().iloc[-1])
        for symbol, df in stock_data
        if not df.empty and not df["Close"].dropna().empty
    }

    all_summaries = "\n---\n".join(summaries)

    from learner import get_learning_context
    learning = await asyncio.to_thread(get_learning_context)

    from news_sentiment import get_sentiment_context
    news_context = await asyncio.to_thread(get_sentiment_context, symbols)
    if not str(news_context).startswith("NEWS SENTIMENT ANALYSIS:"):
        reason = f"Sentiment analysis unavailable; trade decisions blocked ({news_context})"
        logger.error(f"    {reason}")
        return [
            {
                "symbol": sym,
                "signal": "HOLD",
                "confidence": 0.0,
                "reason": reason,
                "price": price_map.get(sym, 0.0),
            }
            for sym, _ in stock_data
        ]

    symbol_list = ", ".join(f'"{s}"' for s in symbols)

    prompt = f"""You are a quantitative trading analyst for NSE Indian stocks. Analyze each stock and return trading signals.

{learning}

{news_context}

STOCKS DATA:
{all_summaries}

PORTFOLIO: Rs.{config.INITIAL_CAPITAL} capital, max {config.MAX_POSITION_SIZE_PCT*100}% per position, {config.STOP_LOSS_PCT*100}% stop loss, {config.TAKE_PROFIT_PCT*100}% take profit.

RULES:
1. Use trade history to avoid repeating losing patterns and favor winning setups.
2. NEWS SENTIMENT is critical — bearish news lowers confidence, bullish news boosts it.
3. BEAR market = cautious BUY signals. Negative company news = SELL or HOLD.

OUTPUT: Return a JSON object with a "signals" key containing an array of exactly {len(symbols)} objects, one per stock.
Example format: {{"signals": [...]}}
Each object MUST have these fields:
- "symbol": string (use exact symbol including .NS suffix: {symbol_list})
- "signal": "BUY" or "SELL" or "HOLD"
- "confidence": number 0.0 to 1.0
- "position_size_pct": number 0.01 to 1.0
- "reason": string (brief explanation under 200 chars)
- "entry_price": number or null
- "stop_loss": number or null
- "target": number or null

Return ONLY valid JSON. No markdown, no explanation.
"""

    try:
        results = await _call_ai_async(prompt, want_json=True)

        normalized = []
        if results:
            # Build lookup tolerant of AI stripping/adding .NS suffix
            by_symbol: dict[str, SignalSchema] = {}
            for item in results:
                sym_name = item.symbol if isinstance(item, SignalSchema) else item.get("symbol", "")
                by_symbol[sym_name] = item
                by_symbol[sym_name.replace(".NS", "")] = item
                if not sym_name.endswith(".NS"):
                    by_symbol[sym_name + ".NS"] = item
            for sym, _df in stock_data:
                raw = by_symbol.get(sym, by_symbol.get(sym.replace(".NS", ""), {}))
                normalized.append(_normalize_signal_record(raw, sym, price_map.get(sym, 0.0)))
            logger.info(f"    Normalized {len(normalized)} signals from AI response")
        else:
            logger.error(f"    No signals could be extracted from AI for {len(stock_data)} stocks")
            # Return HOLD signals so the app at least shows something
            for sym, _df in stock_data:
                normalized.append({
                    "symbol": sym,
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "reason": "AI analysis returned empty response",
                    "price": price_map.get(sym, 0.0),
                })
        return normalized
    except Exception as e:
        logger.error(f"    Batch analysis failed: {e}")
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


async def analyze_watchlist_async(watchlist: list[str] | None = None) -> list[dict]:
    """Analyze all stocks in watchlist using a single batched AI call (Async)."""
    from data_fetcher import get_historical_data

    if watchlist is None:
        watchlist = config.WATCHLIST

    logger.info(f"  [AI-SCAN] Starting scan for {len(watchlist)} symbols...")

    async def fetch_one(symbol):
        try:
            df = await asyncio.to_thread(get_historical_data, symbol, period="60d", interval="1d")
            if not df.empty:
                return (symbol, df)
            logger.warning(f"  [AI-SCAN] {symbol}: empty data from yfinance")
            return None
        except Exception as e:
            logger.warning(f"  [AI-SCAN] {symbol}: fetch failed — {e}")
            return None

    # Fetch data concurrently using asyncio.gather
    tasks = [fetch_one(sym) for sym in watchlist]
    results = await asyncio.gather(*tasks)
    stock_data = [res for res in results if res is not None]

    if not stock_data:
        logger.error(f"  [AI-SCAN] All {len(watchlist)} data fetches returned empty — cannot scan")
        return []

    logger.info(f"  [AI-SCAN] Fetched {len(stock_data)}/{len(watchlist)} stocks, sending to AI...")
    return await analyze_batch_async(stock_data)

def analyze_watchlist(watchlist: list[str] | None = None) -> list[dict]:
    """Sync wrapper for analyze_watchlist_async."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    if loop.is_running():
        # In case it's called from an already running loop (like FastAPI without await)
        # It's better to just raise or handle it. This shouldn't happen if properly refactored.
        return asyncio.run_coroutine_threadsafe(analyze_watchlist_async(watchlist), loop).result()
    else:
        return loop.run_until_complete(analyze_watchlist_async(watchlist))

def get_portfolio_advice(portfolio_summary: dict, signals: list[dict]) -> str:
    """Get AI advice on overall portfolio strategy."""
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
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(_call_ai_async(prompt), loop).result()
        else:
            return loop.run_until_complete(_call_ai_async(prompt))
    except Exception as e:
        return f"Could not get portfolio advice: {e}"
