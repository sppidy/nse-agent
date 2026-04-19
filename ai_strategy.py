"""AI-powered trading strategy — Copilot/Haiku -> Ollama -> OpenRouter -> Groq -> Cloudflare -> Gemini."""

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

# Session-level provider circuit breaker. When a provider returns 402
# ("no credits") or persistently fails, we cool it off so the cascade
# doesn't waste 25-30 seconds per cycle retrying a known-dead endpoint.
import time as _time
_PROVIDER_COOLDOWN_UNTIL: dict[str, float] = {}
_PROVIDER_COOLDOWN_SEC = 30 * 60  # 30 minutes

def _provider_alive(name: str) -> bool:
    until = _PROVIDER_COOLDOWN_UNTIL.get(name, 0)
    return _time.time() >= until

def _mark_provider_down(name: str, seconds: int = _PROVIDER_COOLDOWN_SEC) -> None:
    _PROVIDER_COOLDOWN_UNTIL[name] = _time.time() + seconds
    logger.warning(f"    [CIRCUIT] {name} parked for {seconds//60} min (no credits / persistent failure)")

# Copilot proxy (GitHub Copilot -> OpenAI-compatible API on localhost)
_COPILOT_PROXY_URL = os.getenv("COPILOT_PROXY_URL", "http://localhost:4141/v1")
_COPILOT_MODELS = [
    "claude-haiku-4.5",    # Fast, smart, free via Copilot
    "gpt-4o-mini",         # Cheap fallback
]

# Ollama (self-hosted on self-hosted, OpenAI-compatible)
_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://BACKEND_HOST:11434/v1")
_OLLAMA_MODELS = [
    os.getenv("OLLAMA_MODEL", "gemma4:e4b"),
]

# OpenRouter (OpenAI-compatible, many free/cheap models)
_OPENROUTER_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_MODELS = [
    "anthropic/claude-haiku-4.5",          # Smart, cheap
    "meta-llama/llama-3.3-70b-instruct",   # Free on OpenRouter
    "mistralai/mistral-small-3.1-24b-instruct",  # Free
    "google/gemma-3-27b-it",               # Free
]

# Groq models — ordered by capability, diversified to spread rate limits
# Each model has its own TPM/RPM quota on free tier, so more models = more headroom
_GROQ_MODELS = [
    "llama-3.3-70b-versatile",      # Best quality, 6K TPM free
    "qwen/qwen3-32b",               # Good quality, 6K TPM free
    "meta-llama/llama-4-scout-17b-16e-instruct",  # Fast, 6K TPM free
    "mistral-saba-24b",             # Mistral quality, 6K TPM free
    "llama-3.1-8b-instant",          # Fastest fallback, 14.4K RPD
]

# Cloudflare Workers AI (OpenAI-compatible, 10K neurons/day free)
_CLOUDFLARE_ACCOUNT_ID = "809387d76a2179b664ffa0d0fe703719"
_CLOUDFLARE_URL = f"https://api.cloudflare.com/client/v4/accounts/{_CLOUDFLARE_ACCOUNT_ID}/ai/v1"
_CLOUDFLARE_MODELS = [
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",  # Best quality, function calling
    "@cf/qwen/qwen3-30b-a3b-fp8",                # Strong reasoning
    "@cf/mistralai/mistral-small-3.1-24b-instruct",  # Reliable structured output
    "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",  # Good analysis
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
        position_size_pct = float(raw.get("position_size_pct", 0.05))
    except (TypeError, ValueError):
        position_size_pct = 0.05
    position_size_pct = max(0.01, min(position_size_pct, 0.10))  # Hard cap at 10%

    def _safe_num(v):
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    from datetime import datetime, timezone
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
        "generated_at": datetime.now(timezone.utc).isoformat(),
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


async def _call_cloudflare_async(prompt: str, retries: int = _MAX_RETRIES, want_json: bool = False) -> str | list[SignalSchema]:
    """Call Cloudflare Workers AI (OpenAI-compatible)."""
    import httpx

    api_token = os.getenv("CLOUDFLARE_API_TOKEN")
    if not api_token:
        raise Exception("CLOUDFLARE_API_TOKEN not set")

    for model in _CLOUDFLARE_MODELS:
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
                        f"{_CLOUDFLARE_URL}/chat/completions",
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_token}",
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()

                text = data["choices"][0]["message"]["content"].strip()

                if want_json:
                    signals = _parse_signals_from_text(text)
                    if signals:
                        logger.info(f"    Cloudflare/{model}: parsed {len(signals)} signals")
                        return signals
                    logger.warning(f"    Cloudflare/{model}: didn't yield signals, trying next model")
                    break
                return text
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate" in error_str.lower() or "neuron" in error_str.lower():
                    if attempt < retries - 1:
                        wait = _MIN_DELAY_BETWEEN_CALLS * (attempt + 2)
                        logger.warning(f"    Cloudflare/{model} rate limited, waiting {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        logger.warning(f"    Cloudflare/{model} exhausted retries, trying next model...")
                        break
                elif "404" in error_str or "not_found" in error_str.lower():
                    logger.warning(f"    Cloudflare/{model} not found, trying next model...")
                    break
                else:
                    logger.warning(f"    Cloudflare/{model} error: {e}")
                    if attempt < retries - 1:
                        await asyncio.sleep(_MIN_DELAY_BETWEEN_CALLS)
                    else:
                        break
    raise Exception("All Cloudflare models exhausted")


async def _call_copilot_async(prompt: str, retries: int = _MAX_RETRIES, want_json: bool = False) -> str | list[SignalSchema]:
    """Call Claude Haiku via Copilot proxy (OpenAI-compatible)."""
    import httpx

    for model in _COPILOT_MODELS:
        for attempt in range(retries):
            try:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 1500,
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
                    _mark_provider_down("copilot")
                    raise Exception("Copilot proxy not running")
                elif "402" in error_str or "insufficient" in error_str.lower():
                    # No credits — moving on to the next model is pointless
                    # if every Copilot model goes through the same proxy quota.
                    logger.warning(f"    Copilot/{model} no credits (402), parking provider")
                    _mark_provider_down("copilot")
                    raise Exception("Copilot proxy out of credits")
                else:
                    logger.warning(f"    Copilot/{model} error: {e}")
                    if attempt < retries - 1:
                        await asyncio.sleep(_MIN_DELAY_BETWEEN_CALLS)
                    else:
                        break
    raise Exception("All Copilot models exhausted")


async def _call_ollama_async(prompt: str, retries: int = _MAX_RETRIES, want_json: bool = False) -> str | list[SignalSchema]:
    """Call Ollama (OpenAI-compatible) on self-hosted server."""
    import httpx

    for model in _OLLAMA_MODELS:
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
                        f"{_OLLAMA_BASE_URL}/chat/completions",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    resp.raise_for_status()
                    data = resp.json()

                text = data["choices"][0]["message"]["content"].strip()

                if want_json:
                    signals = _parse_signals_from_text(text)
                    if signals:
                        logger.info(f"    Ollama/{model}: parsed {len(signals)} signals")
                        return signals
                    logger.warning(f"    Ollama/{model}: didn't yield signals, trying next model")
                    break
                return text
            except Exception as e:
                error_str = str(e)
                if "connect" in error_str.lower() or "refused" in error_str.lower() or "timeout" in error_str.lower():
                    logger.warning(f"    Ollama unavailable: {e}")
                    _mark_provider_down("ollama")
                    raise Exception("Ollama server not reachable")
                elif "404" in error_str or "not_found" in error_str.lower():
                    logger.warning(f"    Ollama/{model} not found, trying next model...")
                    break
                else:
                    logger.warning(f"    Ollama/{model} error: {e}")
                    if attempt < retries - 1:
                        await asyncio.sleep(_MIN_DELAY_BETWEEN_CALLS)
                    else:
                        break
    raise Exception("All Ollama models exhausted")


async def _call_openrouter_async(prompt: str, retries: int = _MAX_RETRIES, want_json: bool = False) -> str | list[SignalSchema]:
    """Call OpenRouter API (OpenAI-compatible)."""
    import httpx

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise Exception("OPENROUTER_API_KEY not set")

    for model in _OPENROUTER_MODELS:
        for attempt in range(retries):
            try:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 1500,
                }
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(
                        f"{_OPENROUTER_URL}/chat/completions",
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}",
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()

                text = data["choices"][0]["message"]["content"].strip()

                if want_json:
                    signals = _parse_signals_from_text(text)
                    if signals:
                        logger.info(f"    OpenRouter/{model}: parsed {len(signals)} signals")
                        return signals
                    logger.warning(f"    OpenRouter/{model}: didn't yield signals, trying next model")
                    break
                return text
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate" in error_str.lower():
                    if attempt < retries - 1:
                        wait = _MIN_DELAY_BETWEEN_CALLS * (attempt + 2)
                        logger.warning(f"    OpenRouter/{model} rate limited, waiting {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        logger.warning(f"    OpenRouter/{model} exhausted retries, trying next model...")
                        break
                elif "402" in error_str or "insufficient" in error_str.lower():
                    logger.warning(f"    OpenRouter/{model} no credits, trying next model...")
                    break
                elif "404" in error_str or "not_found" in error_str.lower():
                    logger.warning(f"    OpenRouter/{model} not found, trying next model...")
                    break
                else:
                    logger.warning(f"    OpenRouter/{model} error: {e}")
                    if attempt < retries - 1:
                        await asyncio.sleep(_MIN_DELAY_BETWEEN_CALLS)
                    else:
                        break
    raise Exception("All OpenRouter models exhausted")


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
    """Copilot/Haiku -> Ollama -> OpenRouter -> Groq -> Cloudflare -> Gemini."""
    # 1. Copilot proxy (Claude Haiku via GitHub Copilot — free, smart)
    if _provider_alive("copilot"):
        try:
            return await _call_copilot_async(prompt, want_json=want_json)
        except Exception as e:
            logger.warning(f"    Copilot failed, trying Ollama: {e}")
    else:
        logger.info("    [CIRCUIT] Copilot in cooldown — skipping straight to Ollama")

    # 2. Ollama (self-hosted on self-hosted, free)
    if _provider_alive("ollama"):
        try:
            return await _call_ollama_async(prompt, want_json=want_json)
        except Exception as e:
            logger.warning(f"    Ollama failed, trying OpenRouter: {e}")
    else:
        logger.info("    [CIRCUIT] Ollama in cooldown — skipping to OpenRouter")

    # 3. OpenRouter (many models, cheap/free)
    if os.getenv("OPENROUTER_API_KEY") and _provider_alive("openrouter"):
        try:
            return await _call_openrouter_async(prompt, want_json=want_json)
        except Exception as e:
            logger.warning(f"    OpenRouter failed, trying Groq: {e}")
    elif not _provider_alive("openrouter"):
        logger.info("    [CIRCUIT] OpenRouter in cooldown — skipping to Groq")

    # 4. Groq (free, fast)
    if os.getenv("GROQ_API_KEY"):
        try:
            return await _call_groq_async(prompt, want_json=want_json)
        except Exception as e:
            logger.warning(f"    Groq failed, trying Cloudflare: {e}")

    # 5. Cloudflare Workers AI (10K neurons/day free)
    if os.getenv("CLOUDFLARE_API_TOKEN"):
        try:
            return await _call_cloudflare_async(prompt, want_json=want_json)
        except Exception as e:
            logger.warning(f"    Cloudflare failed, falling back to Gemini: {e}")

    # 6. Gemini (final fallback)
    gemini_client = _get_gemini_client()
    if gemini_client:
        schema = list[SignalSchema] if want_json else None
        return await _call_gemini_async(gemini_client, prompt, response_schema=schema)

    raise Exception("No AI provider available.")


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
    """Convert stock data into a rich text summary for AI with key indicators."""
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

    # MACD data
    macd_diff = latest.get("macd_diff")
    prev_macd_diff = prev.get("macd_diff")
    macd_cross = ""
    if pd.notna(macd_diff) and pd.notna(prev_macd_diff):
        if prev_macd_diff <= 0 < macd_diff:
            macd_cross = "BULLISH CROSS"
        elif prev_macd_diff >= 0 > macd_diff:
            macd_cross = "BEARISH CROSS"
        elif macd_diff > 0:
            macd_cross = "positive"
        else:
            macd_cross = "negative"

    # Bollinger Band position
    bb_high = latest.get("bb_high")
    bb_low = latest.get("bb_low")
    bb_pos = ""
    if pd.notna(bb_high) and pd.notna(bb_low) and (bb_high - bb_low) > 0:
        bb_pct = (price - bb_low) / (bb_high - bb_low) * 100
        bb_pos = f"{bb_pct:.0f}% (near {'upper' if bb_pct > 80 else 'lower' if bb_pct < 20 else 'mid'})"

    # RSI divergence check
    rsi_div = ""
    if len(df) >= 10:
        price_5d_ago = float(df["Close"].iloc[-5])
        rsi_5d_ago = df["rsi"].iloc[-5] if pd.notna(df["rsi"].iloc[-5]) else None
        rsi_now = latest["rsi"] if pd.notna(latest.get("rsi")) else None
        if rsi_5d_ago and rsi_now:
            if price > price_5d_ago and rsi_now < rsi_5d_ago:
                rsi_div = "BEARISH DIVERGENCE (price up, RSI down)"
            elif price < price_5d_ago and rsi_now > rsi_5d_ago:
                rsi_div = "BULLISH DIVERGENCE (price down, RSI up)"

    # Support/resistance from recent range
    if len(df) >= 20:
        high_20d = float(df["High"].tail(20).max())
        low_20d = float(df["Low"].tail(20).min())
        dist_to_resist = ((high_20d - price) / price) * 100
        dist_to_support = ((price - low_20d) / price) * 100
    else:
        high_20d = high_5d
        low_20d = low_5d
        dist_to_resist = dist_to_support = 0

    # Up/down day count
    closes_5d = df["Close"].tail(5)
    up_days = int((closes_5d.diff().dropna() > 0).sum())

    summary = (
        f"Symbol: {symbol}\n"
        f"Price: Rs.{price:.2f} ({change_pct:+.2f}%)\n"
        f"5D Range: Rs.{low_5d:.2f}-{high_5d:.2f} | 20D Range: Rs.{low_20d:.2f}-{high_20d:.2f}\n"
        f"Dist to 20D resistance: {dist_to_resist:.1f}% | Dist to 20D support: {dist_to_support:.1f}%\n"
        f"20D Trend: {trend_20d:+.2f}% | 5D direction: {up_days}/5 up days\n"
        f"RSI(14): {latest['rsi']:.1f}"
    )
    if rsi_div:
        summary += f" *** {rsi_div} ***"
    summary += (
        f"\nEMA9: {latest['ema_short']:.2f}, EMA21: {latest['ema_long']:.2f} "
        f"({'Bullish' if latest['ema_short'] > latest['ema_long'] else 'Bearish'})\n"
        f"MACD histogram: {macd_diff:.4f} ({macd_cross})\n"
    )
    if bb_pos:
        summary += f"Bollinger Band position: {bb_pos}\n"
    summary += f"Volatility: {volatility:.2f}%, Vol: {vol_ratio:.1f}x avg\n"

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
    try:
        news_context = await asyncio.wait_for(
            asyncio.to_thread(get_sentiment_context, symbols),
            timeout=15.0,
        )
    except asyncio.TimeoutError:
        logger.warning("    [NEWS] Sentiment fetch timed out (>15s); proceeding with neutral sentiment.")
        news_context = "NEWS SENTIMENT ANALYSIS: Timed out — treat all symbols as NEUTRAL sentiment."
    except Exception as exc:
        logger.warning(f"    [NEWS] Sentiment fetch failed ({exc}); proceeding with neutral sentiment.")
        news_context = "NEWS SENTIMENT ANALYSIS: Unavailable — treat all symbols as NEUTRAL sentiment."

    if not str(news_context).startswith("NEWS SENTIMENT ANALYSIS:"):
        # Still malformed — fall back to neutral rather than blocking the whole cycle.
        logger.warning(f"    [NEWS] Unexpected sentiment payload; treating as NEUTRAL. Raw: {str(news_context)[:120]}")
        news_context = "NEWS SENTIMENT ANALYSIS: Unavailable — treat all symbols as NEUTRAL sentiment."

    symbol_list = ", ".join(f'"{s}"' for s in symbols)

    prompt = f"""NSE intraday signals. Be decisive — return BUY or SELL when 2+ indicators confirm. HOLD only when truly mixed.

{learning}

{news_context}

STOCKS:
{all_summaries}

RULES (terse):
- BUY when 2+ of: RSI<40 oversold rebound, EMA9>EMA21 cross, MACD bullish, volume>1.0x avg, near 20D support.
- SELL when 2+ of: RSI>70 overbought, EMA9<EMA21 cross, MACD bearish, breakdown below support.
- Skip BUY if within 1.5% of 20D resistance. Skip SELL if within 1.5% of 20D support.
- Bearish news = lean HOLD/SELL.
- Position size 0.03–0.08; never >0.10.

OUTPUT: ONE JSON object {{"signals": [...]}}, exactly {len(symbols)} entries, one per symbol in: {symbol_list}.
Fields per entry: symbol, signal (BUY/SELL/HOLD), confidence (0.0–1.0), position_size_pct (0.01–0.10), reason (≤12 words), entry_price, stop_loss, target.
NO markdown, NO commentary. Just the JSON.
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
