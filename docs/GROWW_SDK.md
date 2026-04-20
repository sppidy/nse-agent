# Groww Python SDK — Agent Reference

Mirror of [groww.in/trade-api/docs/python-sdk](https://groww.in/trade-api/docs/python-sdk/) synthesized for offline use by future agents working in this repo. Last synced **2026-04-20** against SDK **1.5.0**.

> **This agent does not place real orders.** The Groww order endpoints
> (`place_order`, `modify_order`, `cancel_order`, smart orders) are documented
> here for future reference only. Current code paths are read-only:
> `get_ltp`, `get_historical_candle_data`, `get_available_margin_details`,
> `get_holdings_for_user`. Any change that adds a write endpoint needs
> explicit user approval — the demat account is unfunded but the endpoint
> surface still exists.

---

## 1. Install & prerequisites

```bash
pip install growwapi pyotp
# Upgrade to latest:
pip install --upgrade growwapi
```

Requirements: Python 3.9+, Groww account, active Trading API subscription.

---

## 2. Authentication

Two flows. **We use flow B** (TOTP-based) in `groww_client.py` — see `_generate_access_token`.

### A. API Key + Secret (daily approval required on Groww dashboard)

```python
from growwapi import GrowwAPI

access_token = GrowwAPI.get_access_token(api_key="YOUR_API_KEY", secret="YOUR_API_SECRET")
groww = GrowwAPI(access_token)
```

Body sent to `POST /v1/token/api/access`: `{"key_type": "approval", "checksum": hmac(secret, timestamp), "timestamp": unix}`.

### B. TOTP (what we use)

```python
from growwapi import GrowwAPI
import pyotp

api_key = "YOUR_TOTP_JWT"               # long-lived JWT, role=auth-totp
totp = pyotp.TOTP("BASE32_SECRET").now()  # 6-digit code, 30s window

access_token = GrowwAPI.get_access_token(api_key=api_key, totp=totp)
groww = GrowwAPI(access_token)
```

Body sent to the same endpoint: `{"key_type": "totp", "totp": "123456"}`.
**Gotcha:** codes flip every 30s; Groww rejects codes that tick over mid-validation. `groww_client._generate_access_token` handles this (waits if <3s remain; retries once on failure).

### Client bootstrap

```python
from growwapi import GrowwAPI
groww = GrowwAPI(access_token)  # pass the minted access_token string
```

---

## 3. Core constants (Annexures)

| Group | Constant | Value |
|-------|----------|-------|
| Exchange | `groww.EXCHANGE_NSE` / `_BSE` / `_MCX` | NSE / BSE / MCX |
| Segment | `groww.SEGMENT_CASH` / `_FNO` / `_COMMODITY` | CASH / FNO / COMMODITY |
| Product | `groww.PRODUCT_CNC` / `_MIS` / `_NRML` | Delivery / Intraday / Margin |
| Order type | `groww.ORDER_TYPE_LIMIT` / `_MARKET` / `_STOP_LOSS` / `_STOP_LOSS_MARKET` | LIMIT / MARKET / SL / SL_M |
| Txn | `groww.TRANSACTION_TYPE_BUY` / `_SELL` | BUY / SELL |
| Validity | `groww.VALIDITY_DAY` | DAY |
| Candle | `groww.CANDLE_INTERVAL_MIN_1` … `_MONTH` | 1m / 5m / 10m / 1h / 4h / 1d / 1w / 1mo |
| Instrument types | | `EQ`, `IDX`, `FUT`, `CE`, `PE` |

Symbol format for `exchange_trading_symbols` (LTP/OHLC): `"<EXCHANGE>_<TRADING_SYMBOL>"` — e.g. `"NSE_RELIANCE"`.

Groww slug format (used in some fundamentals/MCP responses): lowercase hyphen-joined — e.g. `"reliance-industries-ltd"`.

---

## 3b. Rate limits (free tier — all we have)

Documented at [groww.in/trade-api/docs](https://groww.in/trade-api/docs). Limits are **per type**, not per endpoint — if any endpoint under a type is exhausted, the whole type rate-limits.

| Type | Endpoints | Per second | Per minute |
|------|-----------|-----------|------------|
| **Live Data** | `get_ltp`, `get_ohlc`, `get_quote`, historical candles | **10** | **300** |
| Orders | `place_order`, `modify_order`, `cancel_order` | 10 | 250 |
| Non Trading | order list/status, positions, holdings, margin | 20 | 500 |

### What we enforce

`groww_client._live_data_limiter` is a sliding-window limiter capped at **8 rps / 250 rpm** (20% below Groww's Live-Data cap). Every call to `_GROWW_BASE/v1/live-data/*` or `/v1/historical/*` goes through `_api_get` which:
1. Calls `_live_data_limiter.acquire()` — blocks until a slot is free.
2. Issues the request.
3. If a 429 comes back anyway (e.g., another process sharing credentials), honors `Retry-After` and retries once.

**Result:** the autopilot physically cannot exceed 8 rps against Groww, so we never trip 429 in the first place. The 429 retry is belt-and-suspenders.

### Per-process limiter caveat

The limiter is in-memory per Python process. If the backend and the autopilot both hammer Groww in the same second, each is capped at 8 rps so they could collectively reach 16 rps — still under Groww's 10 rps. The 20% buffer specifically absorbs this. If we ever run three+ processes against shared creds, upgrade to a Redis-backed limiter.

---

## 4. Live data

| Method | Cap | Returns |
|--------|-----|---------|
| `groww.get_ltp(segment, exchange_trading_symbols)` | **50 symbols/call** | `{"NSE_RELIANCE": 2500.5, ...}` |
| `groww.get_ohlc(segment, exchange_trading_symbols)` | **50 symbols/call** | `{symbol: {open, high, low, close}, ...}` |
| `groww.get_quote(exchange, segment, trading_symbol)` | 1 symbol | 30+ fields: prices, bid/ask, OHLC, Greeks, market cap, volume, 52w range |
| `groww.get_option_chain(exchange, underlying, expiry_date)` | 1 chain | Underlying LTP + strike-wise CE/PE with Greeks (delta/gamma/theta/vega/rho/iv), LTP, OI, volume |
| `groww.get_greeks(exchange, underlying, trading_symbol, expiry)` | 1 contract | `{delta, gamma, theta, vega, rho, iv}` |

```python
# LTP example
groww.get_ltp(segment=groww.SEGMENT_CASH, exchange_trading_symbols="NSE_RELIANCE")
# → {"NSE_RELIANCE": 2500.5}

# Quote (richest single-symbol call)
groww.get_quote(exchange=groww.EXCHANGE_NSE, segment=groww.SEGMENT_CASH, trading_symbol="NIFTY")
```

Supported exchanges: NSE, BSE. Segments for live data: CASH, FNO.

---

## 5. Historical data

```python
groww.get_historical_candle_data(
    trading_symbol="RELIANCE",
    exchange=groww.EXCHANGE_NSE,
    segment=groww.SEGMENT_CASH,
    start_time="2026-04-01 09:15:00",   # or epoch ms
    end_time="2026-04-20 15:30:00",
    interval_in_minutes=5,              # optional, default 5
)
# → candles: [[ts_epoch_sec, open, high, low, close, volume], ...]
```

**Deprecated** in favor of the newer `get_historical_candles` (same shape, uses `groww_symbol` like `"NSE-WIPRO"` and `candle_interval` enum).

### Intervals & limits

| Interval | Max range per call | Historical depth |
|----------|--------------------|------------------|
| 1 min | 7 days | Last 3 months |
| 5 min | 15 days | Last 3 months |
| 10 min | 30 days | Last 3 months |
| 1 hour | 150 days | Last 3 months |
| 4 hours | 365 days | Last 3 months |
| 1 day | 1080 days (~3y) | Full history |
| 1 week | No limit | Full history |

> **Our use**: `data_fetcher.get_historical_data` and `get_intraday_data` hit `/v1/historical/candle/range` directly (not via SDK) — the SDK's method wraps the same endpoint.

---

## 6. Instruments (symbol lookup)

| Method | Use case |
|--------|----------|
| `get_instrument_by_groww_symbol("NSE-RELIANCE")` | Lookup by Groww-prefixed symbol |
| `get_instrument_by_exchange_and_trading_symbol(exchange, trading_symbol)` | Standard lookup |
| `get_instrument_by_exchange_token("2885")` | Lookup by exchange-assigned token |
| `get_all_instruments()` | **Full instruments DataFrame** (all tradable symbols) |

```python
sel = groww.get_instrument_by_groww_symbol("NSE-RELIANCE")
# → {"trading_symbol": "RELIANCE", "segment": "CASH", "exchange_token": "2885", ...}

all_instruments = groww.get_all_instruments()  # pandas DataFrame
```

Useful for bootstrapping our scan pool or resolving ambiguous names.

---

## 7. Orders (NOT USED — read carefully before enabling)

> Write endpoints. Anything below mutates broker state. Agent stays in paper-trade mode.

### Place order

```python
groww.place_order(
    trading_symbol="WIPRO",
    quantity=1,
    validity=groww.VALIDITY_DAY,
    exchange=groww.EXCHANGE_NSE,
    segment=groww.SEGMENT_CASH,
    product=groww.PRODUCT_CNC,
    order_type=groww.ORDER_TYPE_LIMIT,
    transaction_type=groww.TRANSACTION_TYPE_BUY,
    price=250,
    trigger_price=245,               # only for SL / SL_M
    order_reference_id="uniq-id",   # client-side dedup
)
```

### Modify / cancel / inspect

```python
groww.modify_order(quantity, order_type, segment, groww_order_id, price=None, trigger_price=None)
groww.cancel_order(segment, groww_order_id)
groww.get_order_status(groww_order_id, segment)
groww.get_order_status_by_reference(order_reference_id, segment)
groww.get_order_detail(groww_order_id, segment)
groww.get_order_list(segment=None, page=0, page_size=100)
groww.get_trade_list_for_order(groww_order_id, segment, page=0, page_size=50)
```

---

## 8. Smart Orders — GTT & OCO (NOT USED)

- **GTT (Good-Till-Triggered):** single conditional order, fires when price crosses a threshold.
- **OCO (One-Cancels-Other):** paired target + stop-loss; execution of one cancels the other.

### Create GTT

```python
groww.create_smart_order(
    smart_order_type=groww.SMART_ORDER_TYPE_GTT,
    reference_id="gtt-ref-unique123",
    segment=groww.SEGMENT_CASH,
    trading_symbol="TCS",
    quantity=10,
    product_type=groww.PRODUCT_CNC,
    exchange=groww.EXCHANGE_NSE,
    duration=groww.VALIDITY_DAY,
    trigger_price="3985.00",
    trigger_direction=groww.TRIGGER_DIRECTION_DOWN,   # or _UP
    order={
        "order_type": groww.ORDER_TYPE_LIMIT,
        "price": "3990.00",
        "transaction_type": groww.TRANSACTION_TYPE_BUY,
    },
)
```

### Create OCO (target + stop-loss bracket)

```python
groww.create_smart_order(
    smart_order_type=groww.SMART_ORDER_TYPE_OCO,
    reference_id="oco-ref-unique456",
    segment=groww.SEGMENT_FNO,
    trading_symbol="NIFTY25OCT24000CE",
    quantity=50,
    product_type=groww.PRODUCT_MIS,
    exchange=groww.EXCHANGE_NSE,
    duration=groww.VALIDITY_DAY,
    net_position_quantity=50,
    transaction_type=groww.TRANSACTION_TYPE_SELL,
    target={"trigger_price": "120.50", "order_type": groww.ORDER_TYPE_LIMIT, "price": "121.00"},
    stop_loss={"trigger_price": "95.00", "order_type": groww.ORDER_TYPE_STOP_LOSS_MARKET, "price": None},
)
```

### Lifecycle

```python
groww.modify_smart_order(smart_order_id, smart_order_type, segment, quantity, trigger_price, trigger_direction, order, duration, product_type, target, stop_loss)
groww.cancel_smart_order(segment, smart_order_type, smart_order_id)
groww.get_smart_order(segment, smart_order_type, smart_order_id)
groww.get_smart_order_list(segment, smart_order_type, status, page, page_size, start_date_time, end_date_time)
```

### Smart-order constants

```python
groww.SMART_ORDER_TYPE_GTT        # "GTT"
groww.SMART_ORDER_TYPE_OCO        # "OCO"
groww.TRIGGER_DIRECTION_UP        # "UP"
groww.TRIGGER_DIRECTION_DOWN      # "DOWN"
groww.SMART_ORDER_STATUS_ACTIVE
groww.SMART_ORDER_STATUS_TRIGGERED
groww.SMART_ORDER_STATUS_CANCELLED
groww.SMART_ORDER_STATUS_EXPIRED
groww.SMART_ORDER_STATUS_FAILED
groww.SMART_ORDER_STATUS_COMPLETED
```

---

## 9. Portfolio

```python
# All holdings (delivery, DP)
groww.get_holdings_for_user(timeout=5)
# → fields: isin, trading_symbol, quantity, average_price, pledge_quantity,
#           demat_locked_quantity, t1_quantity, demat_free_quantity, ...

# All positions (intraday + F&O)
groww.get_positions_for_user(segment=None)  # or SEGMENT_CASH / _FNO / _COMMODITY
# → fields: trading_symbol, segment, credit/debit quantities + prices,
#           carry-forward data, exchange, isin, net_quantity, product, realised_pnl

# Single symbol
groww.get_position_for_trading_symbol(trading_symbol="RELIANCE", segment=groww.SEGMENT_CASH)
```

---

## 10. Margin (READ-ONLY — we use #10.1 already)

### 10.1 Available margin — `get_available_margin_details()`

Key fields in the response:
- `clear_cash` (float)
- `net_margin_used` (float)
- `brokerage_and_charges` (float)
- `collateral_used`, `collateral_available`, `adhoc_margin`
- `fno_margin_details`: `{net_fno_margin_used, span_margin_used, exposure_margin_used, future_balance_available, option_buy_balance_available, option_sell_balance_available}`
- `equity_margin_details`: `{net_equity_margin_used, cnc_margin_used, mis_margin_used, cnc_balance_available, mis_balance_available}`
- `commodity_margin_details`: SPAN / exposure / tender / special / additional / unrealised_m2m / realised_m2m

### 10.2 Pre-check margin for a hypothetical order

```python
groww.get_order_margin_details(
    segment=groww.SEGMENT_CASH,
    orders=[{
        "trading_symbol": "RELIANCE",
        "transaction_type": groww.TRANSACTION_TYPE_BUY,
        "quantity": 1,
        "price": 2500,
        "order_type": groww.ORDER_TYPE_LIMIT,
        "product": groww.PRODUCT_CNC,
        "exchange": groww.EXCHANGE_NSE,
    }],
)
# → {exposure_required, span_required, option_buy_premium, brokerage_and_charges, total_requirement, cash_cnc_margin_required}
```

---

## 11. Feed (WebSocket real-time — NOT USED YET)

Alternative to polling `get_ltp`. NATS-backed; one blocking consumer per process.

```python
from growwapi import GrowwFeed, GrowwAPI

groww = GrowwAPI(access_token)
feed = GrowwFeed(groww)

instruments = [
    {"exchange": "NSE", "segment": "CASH", "exchange_token": "2885"},   # RELIANCE
    {"exchange": "NSE", "segment": "FNO",  "exchange_token": "35241"},
]

def on_data(meta):
    # meta = {exchange, segment, feed_type, feed_key}
    print(feed.get_ltp())

feed.subscribe_ltp(instruments, on_data_received=on_data)
feed.consume()    # BLOCKING — nothing below this line runs
```

### Feed methods

| Capability | Subscribe | Getter | Unsubscribe |
|-----------|-----------|--------|-------------|
| LTP | `subscribe_ltp(list, on_data=…)` | `get_ltp()` | `unsubscribe_ltp(list)` |
| Index value | `subscribe_index_value(list, on_data=…)` | `get_index_value()` | `unsubscribe_index_value(list)` |
| Market depth | `subscribe_market_depth(list, on_data=…)` | `get_market_depth()` | `unsubscribe_market_depth(list)` |
| F&O order updates | `subscribe_fno_order_updates(on_data=…)` | `get_fno_order_update()` | `unsubscribe_fno_order_updates()` |
| Equity order updates | `subscribe_equity_order_updates(on_data=…)` | `get_equity_order_update()` | `unsubscribe_equity_order_updates()` |
| F&O position updates | `subscribe_fno_position_updates(on_data=…)` | `get_fno_position_update()` | `unsubscribe_fno_position_updates()` |

Synchronous polling mode (no blocking consume):

```python
feed.subscribe_ltp(instruments)
for _ in range(10):
    time.sleep(3)
    print(feed.get_ltp())
feed.unsubscribe_ltp(instruments)
```

### Would this help us?

- **Potentially yes** for the autopilot watchlist: one persistent feed replaces N LTP REST calls per cycle.
- **But:** `feed.consume()` blocks the thread — integrating into autopilot's existing sync loop needs a dedicated reader thread + a thread-safe last-price cache. Non-trivial. Defer until we actually hit REST rate limits.

---

## 12. User

```python
groww.get_user_profile()
# → {
#   vendor_user_id,    # our unique id
#   ucc,               # Unique Client Code
#   nse_enabled: bool,
#   bse_enabled: bool,
#   ddpi_enabled: bool,  # Demat Debit and Pledge Instruction (broker-initiated debits)
#   active_segments: [...],
# }
```

---

## 13. Exceptions

Located in `growwapi.groww.exceptions`:

| Class | When raised |
|-------|-------------|
| `GrowwBaseException(msg)` | Catch-all for non-categorized errors |
| `GrowwAPIException(msg, code)` | Parent of all HTTP-error exceptions |
| `GrowwAPIAuthenticationException` | Token invalid/expired, TOTP wrong |
| `GrowwAPIAuthorisationException` | Insufficient permissions for the operation |
| `GrowwAPIBadRequestException` | Malformed body / invalid params |
| `GrowwAPINotFoundException` | Resource doesn't exist |
| `GrowwAPIRateLimitException` | Too many calls |
| `GrowwAPITimeoutException` | Response timeout |
| `GrowwFeedException(msg)` | Generic feed errors |
| `GrowwFeedConnectionException(msg)` | NATS connection failures |
| `GrowwFeedNotSubscribedException(msg, topic)` | Reading from unsubscribed stream |

HTTP status → exception class mapping isn't documented; treat the status `code` attribute on `GrowwAPIException` as ground truth for retry logic.

---

## 14. Changelog (most recent 5 releases)

| Version | Date | Notes |
|---------|------|-------|
| **1.5.0** | 2025-12-03 | MCX commodity trading |
| **1.4.0** | 2025-12-02 | `get_user_profile` added |
| **1.3.0** | 2025-11-24 | Option chain API added |
| **1.2.0** | 2025-10-29 | Smart Orders (GTT + OCO) launched |
| **1.1.0** | 2025-10-08 | NATS ping stability improvements |

---

## 15. How our agent uses the SDK today

| Area | SDK surface | Our code | Notes |
|------|-------------|----------|-------|
| Auth (token minting) | `GrowwAPI.get_access_token(api_key, totp=…)` | `groww_client._generate_access_token` | Direct SDK call. TOTP-window race handled with retry. |
| Live price | `groww.get_ltp` | `groww_client.fetch_live_price` | Hits REST directly (`/v1/live-data/ltp`) rather than via SDK — simpler cache. |
| Historical | `groww.get_historical_candle_data` | `groww_client.fetch_candles` | Same — direct REST to `/v1/historical/candle/range`. |
| Margin | `groww.get_available_margin_details` | via Groww MCP (`groww_mcp.py`) | Only used for safety check (confirm no funds). |
| Holdings | `groww.get_holdings_for_user` | via Groww MCP | Safety check. |
| Market movers / screener | Not in SDK | via Groww MCP (`groww_mcp.get_market_movers`) | MCP-only feature. |
| Fundamentals | Not in SDK | via Groww MCP (`groww_mcp.get_stock_fundamentals`) | MCP-only feature. |
| Orders (all) | `place_order`, smart orders, etc. | **Not wired.** | Agent stays in paper-trade mode. |
| Feed (WebSocket) | `GrowwFeed` | **Not wired.** | Future optimization if REST rate-limits bite. |

---

## 15b. Backtesting / FnO discovery helpers

Despite the name, these are utilities for resolving FnO contracts (not a backtest runner). The actual historical-data call `get_historical_candles` is the new-style replacement for `get_historical_candle_data` (§5).

```python
# FnO expiry dates for an underlying
groww.get_expiries(
    exchange=groww.EXCHANGE_NSE,
    underlying_symbol="NIFTY",
    year=2026,      # optional: 2020..current
    month=1,        # optional: 1-12
)
# → ["2026-01-29", "2026-02-26", ...]

# Available FnO contracts for a specific expiry
groww.get_contracts(
    exchange=groww.EXCHANGE_NSE,
    underlying_symbol="NIFTY",
    expiry_date="2026-01-29",
)
# → list of Groww symbols (strike-wise CE / PE and futures)

# Historical candles — new API (uses groww_symbol + candle_interval enum)
groww.get_historical_candles(
    exchange=groww.EXCHANGE_NSE,
    segment=groww.SEGMENT_CASH,
    groww_symbol="NSE-WIPRO",
    start_time="2026-01-01 09:15:00",   # or epoch seconds
    end_time="2026-01-31 15:30:00",
    candle_interval=groww.CANDLE_INTERVAL_MIN_30,
)
# → OHLC + volume + open_interest (FnO only)
```

---

## 16. Source URLs (for re-sync)

- Index: https://groww.in/trade-api/docs/python-sdk
- Instruments: https://groww.in/trade-api/docs/python-sdk/instruments
- Orders: https://groww.in/trade-api/docs/python-sdk/orders
- Smart Orders: https://groww.in/trade-api/docs/python-sdk/smart-orders
- Portfolio: https://groww.in/trade-api/docs/python-sdk/portfolio
- Margin: https://groww.in/trade-api/docs/python-sdk/margin
- Live Data: https://groww.in/trade-api/docs/python-sdk/live-data
- Historical Data: https://groww.in/trade-api/docs/python-sdk/historical-data
- Backtesting: https://groww.in/trade-api/docs/python-sdk/backtesting
- Feed: https://groww.in/trade-api/docs/python-sdk/feed
- User: https://groww.in/trade-api/docs/python-sdk/user
- Annexures: https://groww.in/trade-api/docs/python-sdk/annexures
- Exceptions: https://groww.in/trade-api/docs/python-sdk/exceptions
- Changelog: https://groww.in/trade-api/docs/python-sdk/changelog
