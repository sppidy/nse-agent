"""Groww MCP client — sync wrapper for the agent.

Connects to https://mcp.groww.in/mcp/ via the official Python `mcp` package.
Auth reuses the Bearer token minted by `groww_client._token()` (same
GROWW_API_KEY + GROWW_TOTP_SECRET credentials — no separate MCP auth).

Each public helper opens a session for its call and tears it down. For the
agent's cadence (once per 15 min cycle) the per-call connection overhead is
negligible; switching to a persistent session is a future optimization.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from logger import logger

_MCP_URL = os.environ.get("GROWW_MCP_URL", "https://mcp.groww.in/mcp/")


async def _call_tool_async(name: str, arguments: dict) -> Any:
    # Import inside function — the package isn't available in minimal
    # environments (tests, tooling) and we want data_fetcher to stay
    # importable regardless.
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    import groww_client

    token = groww_client._token()
    if not token:
        raise RuntimeError("No Groww access token available")
    headers = {"Authorization": f"Bearer {token}"}

    async with streamablehttp_client(_MCP_URL, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
    if result.isError:
        raise RuntimeError(f"MCP tool {name!r} returned error: {result}")
    if not result.content:
        return None
    text = getattr(result.content[0], "text", None)
    if text is None:
        return result.content[0]
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


def call_tool(name: str, arguments: dict | None = None) -> Any:
    """Sync entry point. Spins up a fresh asyncio loop per call."""
    return asyncio.run(_call_tool_async(name, arguments or {}))


def is_available() -> bool:
    """Cheap check — is the mcp package importable AND do we have a token?"""
    try:
        import mcp  # noqa: F401
        import groww_client
        return bool(groww_client._token())
    except Exception:
        return False


# ─── Typed helpers the agent calls ────────────────────────────────────────


def get_market_movers(filters: list[str] | None = None, size: int = 20) -> list[dict]:
    """Fetch top movers. Returns a flat list of {symbol, name, ltp, day_change_perc, source_filter}.

    Empty list on failure — callers should treat this as "no fresh data"
    and fall back to the static pool rather than clearing the pool.
    """
    if filters is None:
        filters = ["TOP_GAINERS", "VOLUME_SHOCKERS", "POPULAR_STOCKS_INTRADAY_VOLUME"]
    try:
        res = call_tool(
            "fetch_market_movers_and_trending_stocks_funds",
            {"discovery_filter_types": filters, "size": size},
        )
    except Exception as e:
        logger.warning(f"Groww MCP movers failed: {e}")
        return []

    data = ((res or {}).get("result") or {}).get("data") or {}
    seen: set[str] = set()
    out: list[dict] = []
    for filter_key, entries in data.items():
        for entry in entries or []:
            company = entry.get("company") or {}
            stats = entry.get("stats") or {}
            # Movers responses come in two shapes:
            #   (a) {company: {nseScriptCode, companyName, ...}, stats: {ltp, dayChangePerc, ...}}
            #   (b) flat {companyName, ltp, volume, ...}  (e.g., VOLUME_SHOCKERS)
            nse_code = company.get("nseScriptCode") or entry.get("nseScriptCode")
            if not nse_code:
                # VOLUME_SHOCKERS lacks nseScriptCode — skip rather than guess.
                continue
            yf_symbol = nse_code + ".NS"
            if yf_symbol in seen:
                continue
            seen.add(yf_symbol)
            out.append({
                "symbol": yf_symbol,
                "name": company.get("companyName") or entry.get("companyName"),
                "ltp": stats.get("ltp") or entry.get("ltp"),
                "day_change_perc": stats.get("dayChangePerc"),
                "source_filter": filter_key,
            })
    return out


def _strip_suffix(sym: str) -> str:
    return sym.removesuffix(".NS").removesuffix(".BO")


def _parse_cr(value) -> float | None:
    """Parse Groww money strings like '₹18,47,326Cr' to a float in crores."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).replace(",", "").replace("₹", "").replace("Cr", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _parse_percent(value) -> float | None:
    """Parse Groww percent strings like '9.47%' to a float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).replace("%", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _best_match(content: dict, query: str) -> dict | None:
    """Pick the best fundamentals entry for `query` out of Groww's response.

    Groww returns slug-keyed matches (e.g. for query "RELIANCE" it might return
    both reliance-power-ltd and reliance-industries-ltd). Prefer an exact slug
    match (`<query>-ltd` or `<query>-limited`), else the largest market cap.
    """
    q = query.lower().strip()
    if not content:
        return None
    exact_candidates = [q, f"{q}-ltd", f"{q}-limited"]
    for key in exact_candidates:
        if key in content:
            return content[key]
    # Fallback: prefer entry with highest parsed market cap
    def _mcap(entry: dict) -> float:
        stats = entry.get("fundamentals_stats") or {}
        return _parse_cr(stats.get("marketCap")) or 0.0
    sorted_entries = sorted(content.values(), key=_mcap, reverse=True)
    return sorted_entries[0] if sorted_entries else None


def get_stock_fundamentals(symbol: str) -> dict | None:
    """Fetch fundamentals stats for one NSE symbol. None on failure.

    Returns a normalized dict with numeric fields:
        {pe, debt_equity, market_cap_cr, price_to_book, roe_pct, div_yield_pct,
         industry_pe, book_value, eps_ttm}
    Missing fields in Groww's response are simply absent from the result.
    """
    clean = _strip_suffix(symbol)
    try:
        res = call_tool(
            "fetch_stocks_fundamental_data",
            {"company_names": [clean], "view": "stats_only"},
        )
    except Exception as e:
        logger.warning(f"Groww MCP fundamentals failed for {symbol}: {e}")
        return None

    content = ((res or {}).get("result") or {}).get("content") or {}
    entry = _best_match(content, clean)
    if not entry:
        return None
    stats = entry.get("fundamentals_stats") or {}
    normalized: dict = {}
    if "peRatio" in stats:
        normalized["pe"] = stats["peRatio"]
    if "debtToEquity" in stats:
        normalized["debt_equity"] = stats["debtToEquity"]
    if "marketCap" in stats:
        mc = _parse_cr(stats["marketCap"])
        if mc is not None:
            # Store in absolute rupees for symmetry with yfinance's marketCap
            normalized["market_cap"] = mc * 1e7
            normalized["market_cap_cr"] = mc
    if "pbRatio" in stats:
        normalized["price_to_book"] = stats["pbRatio"]
    if "roe" in stats:
        normalized["roe_pct"] = _parse_percent(stats["roe"])
    if "dividendYieldInPercent" in stats:
        normalized["div_yield_pct"] = _parse_percent(stats["dividendYieldInPercent"])
    if "industryPe" in stats:
        normalized["industry_pe"] = stats["industryPe"]
    if "bookValue" in stats:
        normalized["book_value"] = stats["bookValue"]
    if "epsTtm" in stats:
        normalized["eps_ttm"] = stats["epsTtm"]
    return normalized
