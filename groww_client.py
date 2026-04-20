"""Groww REST client — NSE/BSE OHLCV data source.

Drop-in replacement for a subset of yfinance calls. Two auth paths:

1. **Programmatic** (preferred — self-renewing). Set `GROWW_API_KEY` +
   `GROWW_TOTP_SECRET` env vars. We call the growwapi SDK to mint a fresh
   access token on first use and cache it for 23h. TOTP secret is the
   base32 string from Groww's authenticator setup.
2. **Static JWT** (fallback — manual daily rotation). Set `GROWW_ACCESS_TOKEN`
   to a JWT copied from Groww's web session / vendor dashboard. Expires
   daily; re-paste each morning.

The token is IP-restricted to the deploy server (self-hosted-attached). On
any auth/network failure, data_fetcher falls back to yfinance.
"""

import base64
import json
import math
import os
import threading
import time

import pandas as pd
import requests

from logger import logger


_GROWW_BASE = "https://api.groww.in"

_TOKEN_TTL_CAP_SECONDS = 23 * 3600  # cap — real TTL derived from JWT exp
_TOKEN_REFRESH_GRACE_SECONDS = 60   # refresh this many seconds before exp
_AUTH_FAILURE_COOLDOWN = 60.0       # don't retry failed SDK auth for 60s
_token_cache: dict[str, object] = {"token": None, "expires_at": 0.0, "last_failure": 0.0}
_token_lock = threading.Lock()


def _jwt_exp(token: str) -> int | None:
    """Parse the `exp` claim out of a JWT without verifying the signature."""
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        exp = payload.get("exp")
        return int(exp) if exp is not None else None
    except Exception:
        return None


def _generate_access_token() -> str | None:
    """Mint a fresh access token via Groww SDK (TOTP JWT + base32 secret). None on failure.

    Handles TOTP-window race: codes flip every 30s, and Groww rejects codes that
    tick over mid-validation. If we're within 3s of a window boundary we wait
    for the next tick; on any failure we retry once with a fresh code.
    """
    api_key = os.environ.get("GROWW_API_KEY", "").strip()
    totp_secret = os.environ.get("GROWW_TOTP_SECRET", "").strip()
    if not api_key or not totp_secret:
        return None

    try:
        from growwapi import GrowwAPI
        import pyotp
    except Exception as e:
        logger.error(f"Groww SDK import failed: {e}")
        return None

    totp = pyotp.TOTP(totp_secret)

    def _wait_if_window_end() -> None:
        # If <3s remain in the current 30s window, wait for the next one.
        pos = int(time.time()) % 30
        if pos > 27:
            time.sleep(31 - pos)

    def _try_once() -> object:
        _wait_if_window_end()
        return GrowwAPI.get_access_token(api_key, totp=totp.now())

    res: object
    try:
        res = _try_once()
    except Exception as e:
        logger.warning(f"Groww SDK auth 1st attempt failed: {e}; retrying after window tick…")
        try:
            time.sleep(max(1.0, 30 - (time.time() % 30)))
            res = _try_once()
        except Exception as e2:
            logger.error(f"Groww SDK auth failed (after retry): {e2}")
            return None

    # SDK docs say the return is `dict`, but in practice it returns the token
    # string directly. Handle both.
    if isinstance(res, str) and res.strip():
        return res.strip()
    if isinstance(res, dict):
        for key in ("access_token", "accessToken", "token"):
            value = res.get(key)
            if value:
                return str(value)
        logger.error(f"Groww SDK response missing access_token key. Keys: {list(res.keys())}")
        return None
    logger.error(f"Groww SDK returned unexpected type: {type(res).__name__}")
    return None


def _token() -> str:
    """Cached programmatic token if fresh, else mint a new one, else static JWT fallback."""
    now = time.time()
    with _token_lock:
        cached = _token_cache["token"]
        if cached and now < _token_cache["expires_at"]:
            return str(cached)
        if now - _token_cache["last_failure"] < _AUTH_FAILURE_COOLDOWN:
            # Recent SDK failure — skip retry, use static fallback
            return os.environ.get("GROWW_ACCESS_TOKEN", "").strip()
        fresh = _generate_access_token()
        if fresh:
            _token_cache["token"] = fresh
            jwt_exp = _jwt_exp(fresh)
            if jwt_exp:
                # Refresh shortly before the JWT's own exp, but cap at 23h so
                # a mis-parsed / absurd exp doesn't pin a stale token forever.
                _token_cache["expires_at"] = min(jwt_exp - _TOKEN_REFRESH_GRACE_SECONDS, now + _TOKEN_TTL_CAP_SECONDS)
            else:
                _token_cache["expires_at"] = now + _TOKEN_TTL_CAP_SECONDS
            logger.info("Groww access token refreshed via SDK (expires in %ds)", int(_token_cache["expires_at"] - now))
            return fresh
        # SDK path failed (or not configured); remember and fall back to static
        _token_cache["last_failure"] = now
    return os.environ.get("GROWW_ACCESS_TOKEN", "").strip()


def is_configured() -> bool:
    return bool(_token())


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_token()}",
        "X-API-VERSION": "1.0",
        "Accept": "application/json",
    }


def _to_groww_symbol(sym: str) -> tuple[str, str, str]:
    """Convert a yfinance-style symbol to (trading_symbol, exchange, segment)."""
    s = sym.upper().strip()
    if s in ("^NSEI", "NIFTY", "NIFTY50", "^NIFTY"):
        return "NIFTY", "NSE", "INDEX"
    if s in ("^NSEBANK", "BANKNIFTY"):
        return "BANKNIFTY", "NSE", "INDEX"
    if s in ("^BSESN", "SENSEX"):
        return "SENSEX", "BSE", "INDEX"
    if s.endswith(".NS"):
        return s[:-3], "NSE", "CASH"
    if s.endswith(".BO"):
        return s[:-3], "BSE", "CASH"
    return s, "NSE", "CASH"


def _period_to_seconds(period: str) -> int:
    p = period.strip().lower()
    try:
        if p.endswith("mo"):
            return int(p[:-2]) * 30 * 86400
        if p.endswith("y"):
            return int(p[:-1]) * 365 * 86400
        if p.endswith("d"):
            return int(p[:-1]) * 86400
        if p.endswith("h"):
            return int(p[:-1]) * 3600
    except ValueError:
        pass
    # Fallback — 60 days matches yfinance's "60d" autopilot default.
    return 60 * 86400


_INTERVAL_MAP: dict[str, int] = {
    "1m": 1, "5m": 5, "10m": 10, "15m": 15, "30m": 30,
    "60m": 60, "1h": 60, "4h": 240,
    "1d": 1440, "d": 1440, "1day": 1440,
}


def _interval_to_minutes(interval: str) -> int:
    return _INTERVAL_MAP.get(interval.lower().strip(), 1440)


def _candles_to_df(candles: list) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    ts = pd.to_datetime(df["ts"], unit="s", utc=True)
    idx = pd.DatetimeIndex(ts).tz_convert("Asia/Kolkata").tz_localize(None)
    idx.name = "Date"
    df.index = idx
    df = df.drop(columns=["ts"])
    df = df.astype({"Open": "float64", "High": "float64", "Low": "float64", "Close": "float64", "Volume": "int64"})
    return df


class GrowwUnsupported(Exception):
    """Raised for inputs Groww's data APIs don't serve (e.g., indices on LTP).

    Distinct from generic network errors so callers can silently skip without
    spamming the log on every cycle."""


def fetch_candles(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """yfinance-shaped DataFrame (Open/High/Low/Close/Volume, tz-naive datetime index)."""
    if not is_configured():
        raise RuntimeError("GROWW_API_KEY not set")
    trading_symbol, exchange, segment = _to_groww_symbol(symbol)
    if segment == "INDEX":
        # Groww's historical candle endpoint also 400s for INDEX. Skip cleanly
        # so callers fall back to yfinance without logging a per-cycle warning.
        raise GrowwUnsupported(f"Groww historical doesn't support INDEX segment ({symbol})")
    now = int(time.time())
    start = now - _period_to_seconds(period)
    params = {
        "exchange": exchange,
        "segment": segment,
        "trading_symbol": trading_symbol,
        "start_time": str(start),
        "end_time": str(now),
        "interval_in_minutes": str(_interval_to_minutes(interval)),
    }
    r = requests.get(f"{_GROWW_BASE}/v1/historical/candle/range", params=params, headers=_headers(), timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "SUCCESS":
        raise RuntimeError(f"Groww non-success for {symbol}: {data.get('error') or data}")
    candles = (data.get("payload") or {}).get("candles") or []
    return _candles_to_df(candles)


def _clean_ltp(value) -> float | None:
    try:
        price = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    if price is None or not math.isfinite(price) or price <= 0:
        return None
    return price


def fetch_live_prices_batch(symbols: list[str]) -> dict[str, float | None]:
    """Batch LTP for many symbols in ~1 request per 50 symbols.

    Returns ``{yf_symbol: price_or_None}``. Symbols in the INDEX segment
    (e.g. ``^NSEI``) are silently omitted so callers can fall them back to
    yfinance individually. Any HTTP error marks the whole chunk as None —
    the caller is responsible for yfinance fallback per symbol.
    """
    if not is_configured():
        raise RuntimeError("GROWW_API_KEY not set")

    # Group by (exchange, segment) so one request only spans a single
    # segment — Groww's LTP endpoint takes a single segment param.
    by_key: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for yf_sym in symbols:
        try:
            trading, exchange, segment = _to_groww_symbol(yf_sym)
        except Exception:
            continue
        if segment == "INDEX":
            continue
        by_key.setdefault((exchange, segment), []).append((yf_sym, trading))

    out: dict[str, float | None] = {}
    for (exchange, segment), items in by_key.items():
        for i in range(0, len(items), 50):
            chunk = items[i : i + 50]
            keys = [f"{exchange}_{t}" for _, t in chunk]
            params = {"segment": segment, "exchange_symbols": ",".join(keys)}
            try:
                r = requests.get(
                    f"{_GROWW_BASE}/v1/live-data/ltp",
                    params=params,
                    headers=_headers(),
                    timeout=15,
                )
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                logger.warning(f"Groww batch LTP chunk of {len(chunk)} failed: {e}")
                for yf_sym, _ in chunk:
                    out.setdefault(yf_sym, None)
                continue
            if data.get("status") != "SUCCESS":
                for yf_sym, _ in chunk:
                    out.setdefault(yf_sym, None)
                continue
            payload = data.get("payload") or {}
            for yf_sym, trading in chunk:
                out[yf_sym] = _clean_ltp(payload.get(f"{exchange}_{trading}"))
    return out


def fetch_live_price(symbol: str) -> float | None:
    if not is_configured():
        raise RuntimeError("GROWW_API_KEY not set")
    trading_symbol, exchange, segment = _to_groww_symbol(symbol)
    if segment == "INDEX":
        # Groww's LTP endpoint returns 400 for INDEX segment. Caller falls back
        # to yfinance silently.
        raise GrowwUnsupported(f"Groww LTP doesn't support INDEX segment ({symbol})")
    key = f"{exchange}_{trading_symbol}"
    params = {"segment": segment, "exchange_symbols": key}
    r = requests.get(f"{_GROWW_BASE}/v1/live-data/ltp", params=params, headers=_headers(), timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "SUCCESS":
        raise RuntimeError(f"Groww non-success for {symbol}: {data.get('error') or data}")
    return _clean_ltp((data.get("payload") or {}).get(key))
