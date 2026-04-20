"""Fundamentals pre-filter — reject structurally broken stocks before AI scan.

Primary source: Groww MCP (`fetch_stocks_fundamental_data`) — clean numeric
fundamentals from Groww's own dataset.
Fallback: yfinance `Ticker.info` — when Groww MCP is unreachable or the symbol
isn't in Groww's fundamentals corpus.

Cache: `data/fundamentals_cache.json` with 7-day TTL. Fundamentals change on
earnings (quarterly) so weekly refresh is plenty — avoids an MCP hit on every
scan cycle.

Gate is conservative by default: only reject stocks with clearly broken
metrics (extreme PE, extreme leverage, penny-stock market cap). Groww reports
debt/equity on a 0-N scale (0.43 for RELIANCE); yfinance reports it ×100
(43 for RELIANCE). The thresholds below use Groww's scale; we divide yfinance
values by 100 at fetch time.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import yfinance as yf

from logger import logger

_CACHE_PATH = Path(__file__).parent / "data" / "fundamentals_cache.json"
_CACHE_TTL_SECONDS = 7 * 86400  # 1 week

# Reject thresholds — cross ANY of these and the stock is filtered out.
MAX_PE = 150.0              # extreme overvaluation / earnings collapse
MAX_DEBT_EQUITY = 4.0       # Groww scale (0-N); RELIANCE=0.43, banks ~1-2
MIN_MARKET_CAP = 5e8        # ₹50 Cr — filter out micro-cap noise
MAX_PRICE_TO_BOOK = 30.0    # extreme asset-less bubble


def _load_cache() -> dict:
    if not _CACHE_PATH.exists():
        return {}
    try:
        with _CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read fundamentals cache: {e}")
        return {}


def _save_cache(cache: dict) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not write fundamentals cache: {e}")


def _fetch_via_groww(symbol: str) -> dict | None:
    """Try Groww MCP first. Returns the normalized dict or None."""
    try:
        import groww_mcp
    except ImportError:
        return None
    try:
        stats = groww_mcp.get_stock_fundamentals(symbol)
    except Exception as e:
        logger.warning(f"Groww MCP fundamentals raised for {symbol}: {e}")
        return None
    if not stats:
        return None
    stats["fetched_at"] = time.time()
    stats["source"] = "groww"
    return stats


def _fetch_via_yfinance(symbol: str) -> dict | None:
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as e:
        logger.warning(f"yfinance info failed for {symbol}: {e}")
        return None
    pe = info.get("trailingPE") or info.get("forwardPE")
    de = info.get("debtToEquity")
    if isinstance(de, (int, float)):
        de = de / 100.0  # yfinance reports ×100 (e.g. 43 = 0.43)
    return {
        "pe": pe,
        "debt_equity": de,
        "market_cap": info.get("marketCap"),
        "price_to_book": info.get("priceToBook"),
        "fetched_at": time.time(),
        "source": "yfinance",
    }


def _fetch_fresh(symbol: str) -> dict | None:
    fresh = _fetch_via_groww(symbol)
    if fresh:
        return fresh
    return _fetch_via_yfinance(symbol)


def get_fundamentals(symbol: str, force_refresh: bool = False) -> dict | None:
    """Return cached fundamentals for a symbol, refreshing if stale."""
    cache = _load_cache()
    entry = cache.get(symbol)
    if entry and not force_refresh:
        age = time.time() - entry.get("fetched_at", 0)
        if age < _CACHE_TTL_SECONDS:
            return entry
    fresh = _fetch_fresh(symbol)
    if fresh:
        cache[symbol] = fresh
        _save_cache(cache)
        return fresh
    return entry  # stale or None


def passes_filter(symbol: str) -> tuple[bool, str | None]:
    """Return (ok, reason). ok=False -> stock should be skipped before AI scan.

    Missing data (e.g., yfinance hiccup) is treated as PASS — we don't want to
    block trading over a data blip.
    """
    f = get_fundamentals(symbol)
    if not f:
        return True, None  # unknown -> pass

    def _is_num(x) -> bool:
        return isinstance(x, (int, float)) and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))

    pe = f.get("pe")
    if _is_num(pe) and pe > MAX_PE:
        return False, f"PE={pe:.0f} > {MAX_PE:.0f}"
    de = f.get("debt_equity")
    if _is_num(de) and de > MAX_DEBT_EQUITY:
        return False, f"D/E={de:.0f} > {MAX_DEBT_EQUITY:.0f}"
    mc = f.get("market_cap")
    if _is_num(mc) and mc < MIN_MARKET_CAP:
        return False, f"MCap={mc/1e7:.0f}Cr < {MIN_MARKET_CAP/1e7:.0f}Cr"
    pb = f.get("price_to_book")
    if _is_num(pb) and pb > MAX_PRICE_TO_BOOK:
        return False, f"P/B={pb:.0f} > {MAX_PRICE_TO_BOOK:.0f}"

    return True, None
