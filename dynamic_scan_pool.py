"""Dynamic scan pool — live Groww movers via MCP, short-TTL in-memory cache.

Calls `fetch_market_movers_and_trending_stocks_funds` through the Groww MCP
server (see `groww_mcp.py`) on demand. The result is cached in-memory for
`CACHE_TTL_SECONDS` so the 15-minute autopilot loop doesn't hammer the MCP
on every cycle. If the call fails we return an empty list — the caller keeps
its static pool untouched.
"""

from __future__ import annotations

import time

from logger import logger

# Cache just long enough to span one autopilot cycle — if the list shifts
# meaningfully intra-cycle, next cycle picks it up.
CACHE_TTL_SECONDS = 10 * 60  # 10 minutes
DEFAULT_FILTERS = [
    "TOP_GAINERS",
    "VOLUME_SHOCKERS",
    "POPULAR_STOCKS_INTRADAY_VOLUME",
    "HIGH_MOMENTUM",
]
DEFAULT_SIZE = 10

_cache: dict[str, object] = {"at": 0.0, "symbols": []}


def load_dynamic_symbols() -> list[str]:
    """Return a list of yfinance-style NSE symbols (e.g. 'TRENT.NS').

    Empty list on any failure or when Groww MCP isn't configured — callers
    are expected to fall back to their static pool, not clear it.
    """
    now = time.time()
    if _cache["symbols"] and (now - float(_cache["at"])) < CACHE_TTL_SECONDS:
        return list(_cache["symbols"])  # type: ignore[arg-type]

    try:
        import groww_mcp
        if not groww_mcp.is_available():
            return []
        movers = groww_mcp.get_market_movers(filters=DEFAULT_FILTERS, size=DEFAULT_SIZE)
    except Exception as e:
        logger.warning(f"Dynamic scan pool fetch failed: {e}")
        return []

    symbols = [m["symbol"] for m in movers if m.get("symbol")]
    _cache["at"] = now
    _cache["symbols"] = symbols
    if symbols:
        logger.info(f"  [SCAN] Dynamic pool (Groww MCP): {len(symbols)} symbols from {len(DEFAULT_FILTERS)} filters")
    return symbols


def merge_scan_pool(static_pool: list[str]) -> list[str]:
    """Merge the hardcoded pool with the Groww movers snapshot.

    Dynamic symbols are prepended (higher priority) but the static pool still
    participates — this keeps coverage stable if the snapshot is empty.
    Duplicates are dropped, order preserved.
    """
    dynamic = load_dynamic_symbols()
    seen: set[str] = set()
    merged: list[str] = []
    for s in dynamic + static_pool:
        if s not in seen:
            seen.add(s)
            merged.append(s)
    return merged
