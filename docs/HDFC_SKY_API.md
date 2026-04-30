# HDFC Sky Open API — Agent Reference

Mirror of [developer.hdfcsky.com/sky-docs](https://developer.hdfcsky.com/sky-docs/docs/intro) synthesized for offline use by future agents working in this repo. Last synced **2026-04-23** against the live Docusaurus docs site (no explicit SDK version published; REST-only).

> **This agent does not place real orders via HDFC Sky.** Nothing in `groww_client.py`, `groww_mcp.py`, the autopilot, or the backend currently talks to HDFC Sky — the account is linked and funded (see user profile memory) but unused by code. This file documents the endpoint surface so that a future wiring (secondary broker, fallback quotes, holdings reconciliation) can be built quickly. Any change that adds a write endpoint (`/orders`, `/orders/kart`, `/event/gtt`, `/basket*`, `/position/convert`) needs explicit user approval.
>
> Unlike Groww (which ships a Python SDK), HDFC Sky is **REST-only**. There is no official `pip` package — integrations must call the HTTPS endpoints directly with `requests`/`httpx` and manage the `access_token` lifecycle in-app.

---

## 1. Install & prerequisites

No SDK package. Use any HTTP client:

```bash
pip install requests  # or httpx
```

Requirements:
- HDFC Sky trading account + demat.
- API Key + API Secret from the HDFC Sky developer portal ([developer.hdfcsky.com](https://developer.hdfcsky.com)).
- A registered `callbackUrl` for the Authorize step (returned in the authorize response; can be `https://<your-host>` for a headless flow).
- Mandatory `User-Agent` header on **every** request (HDFC rejects calls without it — cURL default UA is not accepted; use a browser UA string).

### Base hosts seen in the docs

| Environment | Host |
|-------------|------|
| Production | `https://developer.hdfcsky.com` |
| UAT / sandbox | `https://uat-developer.hdfcsky.com` |
| WebSocket (UAT) | `ws://uat-developer.hdfcsky.com/wsapi/v1/session` |

> The docs interleave prod and UAT hosts in cURL samples — always substitute the host you actually target. Production WebSocket host is not published; raise it with HDFC support.

---

## 2. Authentication

HDFC Sky offers **two flows**. Both end in an `accessToken` that is passed as the `Authorization` header (no `Bearer` prefix — just the raw JWT) on all trading/data calls.

### A. Backend (API-driven) flow — 6 steps

Everything over REST; no browser redirect. Suitable for headless agents (what we'd use).

```
1. GET  /oapi/v1/login                    → token_id
2. POST /oapi/v1/login-channel/validate   → (prompts OTP delivery)
3. PUT  /oapi/v1/otp/validate             → (triggers security-pin question)
4. POST /oapi/v1/twofa/validate           → request_token
5. GET  /oapi/v1/authorise                → (confirms T&C consent)
6. POST /oapi/v1/access-token             → accessToken (JWT)
```

### B. Frontend (redirect) flow

```
1. Browser → https://<host>/oapi/v1/login?api_key=<api_key>
2. User enters credentials + OTP + PIN on HDFC Sky's hosted UI.
3. User accepts T&Cs.
4. HDFC Sky redirects to the merchant's registered callback with `request_token` in the query.
5. POST /oapi/v1/access-token with api_key, request_token, api_secret → accessToken.
```

### Step-by-step endpoint reference (flow A)

#### 2.1 Token ID — `GET /oapi/v1/login`

```
GET https://developer.hdfcsky.com/oapi/v1/login?api_key=<api_key>
Headers: User-Agent: <browser UA>
```

Response:
```json
{ "tokenId": "71de27013f81441a9cc3f311ad8720a9" }
```

#### 2.2 Validate username — `POST /oapi/v1/login-channel/validate`

```
POST /oapi/v1/login-channel/validate?api_key=<api_key>&token_id=<token_id>
Body: { "username": "<user>" }
```

Response (triggers OTP via SMS + email):
```json
{
  "name": "USER",
  "recaptcha": false,
  "loginId": "TESTUSER123",
  "twofa": {},
  "message": "Please enter otp sent on mobile ******1234 and email ID TEST******@GMAIL.COM.",
  "twoFAEnabled": true
}
```

#### 2.3 Validate OTP — `PUT /oapi/v1/otp/validate`

```
PUT /oapi/v1/otp/validate?api_key=<api_key>&token_id=<token_id>
Body: { "otp": "1234" }
```

Response (now prompts for security PIN):
```json
{
  "loginId": "TESTUSER123",
  "twofa": { "questions": [{ "question": "Please enter your pin" }] },
  "twoFAEnabled": true
}
```

#### 2.4 Validate PIN — `POST /oapi/v1/twofa/validate`

```
POST /oapi/v1/twofa/validate?api_key=<api_key>&token_id=<token_id>
Body: { "answer": "1234" }
```

Response:
```json
{
  "requestToken": "2fde50dba25c46f581b9ea83041123b9473394134",
  "termsAndConditions": { "version": 1, "content": [ { "header": "P&L1", "body": "Terms and conditions agreed" } ] },
  "callbackUrl": "https://test123.com",
  "authorised": true
}
```

#### 2.5 Authorize consent — `GET /oapi/v1/authorise`

```
GET /oapi/v1/authorise?api_key=<api_key>&token_id=<token_id>&consent=True&request_token=<request_token>
```

Response:
```json
{ "callbackUrl": "https://test123.com", "requestToken": "a1561247f2184c119a24e1daee0d495c" }
```

> The `requestToken` returned here (post-consent) is what's used in the final step.

#### 2.6 Access Token — `POST /oapi/v1/access-token`

```
POST /oapi/v1/access-token?api_key=<api_key>&request_token=<request_token>
Body: { "apiSecret": "<api_secret>" }
```

Response:
```json
{ "accessToken": "eyJhbGci...<JWT>..." }
```

#### 2.7 Resend OTP (helper) — `POST /oapi/v1/twofa/resend`

```
POST /oapi/v1/twofa/resend?api_key=<api_key>&token_id=<token_id>
```

No body required.

### Token lifetime

Not explicitly documented in the published docs. Empirically HDFC Sky issues a short-lived JWT (check the `exp` claim); expect to re-authenticate daily or when a 401 hits. Store `accessToken` + decoded `exp` and refresh via the full flow before expiry (there is **no refresh-token endpoint** in the docs).

### Required headers on every subsequent call

```http
Authorization: <accessToken>      ← raw JWT, no "Bearer " prefix
User-Agent:    Mozilla/5.0 ...    ← any non-empty browser UA; missing UA → 403
Content-Type:  application/json   ← on POST / PUT / DELETE with body
```

---

## 3. Rate limits

**Not published.** The docs site does not document per-second / per-minute caps for REST endpoints. One explicit WebSocket limit:

- **300 instruments** max per WebSocket connection.
- **3 concurrent WebSocket connections** per `api_key`.

For REST, assume HDFC will return **429** or **503** under pressure; implement exponential backoff and honor `Retry-After` if present. If we ever wire this broker, back-pressure the autopilot the same way we do for Groww (sliding-window limiter starting at 5 rps, raise until we see failures).

---

## 4. Live data

### 4.1 LTP — `PUT /oapi/v1/fetch-ltp`

```
PUT /oapi/v1/fetch-ltp?api_key=<api_key>
Body: { "data": [ { "exchange": "BSE", "token": "542809" } ] }
```

Response:
```json
{
  "data": [ { "prev_close": 0.28, "ltp": 0.29, "exchange": "BSE", "token": 542809 } ],
  "meta": { "statusCode": "OK", "statusMsg": "OK" }
}
```

> Multiple symbols per call — pass as many `{exchange, token}` objects in `data[]`. Page limit not documented; start with ≤50 per call (same as Groww's cap) until a hard limit surfaces.

### 4.2 WebSocket — `ws://.../wsapi/v1/session`

Preferred for live quotes. Multiplexed streaming with per-scrip subscriptions.

Subscribe / unsubscribe message:
```json
{
  "heart_beat": false,
  "subscribe":   [ { "scripId": "BSE_1",     "type": "LTP" },
                   { "scripId": "BSE_16675", "type": "ALL" } ],
  "unSubscribe": [ { "scripId": "BFO_3004",  "type": "GREEK" } ]
}
```

Heartbeat packet (send periodically to keep the connection alive):
```json
{ "heart_beat": true }
```

#### `scripId` format by exchange × segment

| Exchange | Segment | Prefix |
|----------|---------|--------|
| BSE | CM | `BSE_<token>` |
| BSE | FO | `BFO_<token>` |
| BSE | INDEX | `BSE_INDEX_<token>` |
| NSE | CM | `NSE_<token>` |
| NSE | FO | `NFO_<token>` |
| NSE | CD | `NCD_<token>` |
| NSE | INDEX | `NSE_INDEX_<token>` |
| MCX | FO | `MCX_<token>` |

#### Subscription `type`

`LTP`, `ALL`, `GREEK` (and implicitly `INDEX` / `ORDER` / `TRADE` via `ALL`).

#### Packet types (protobuf-encoded frames)

| Packet Type | Payload field |
|-------------|---------------|
| `HEARTBEAT` | only `packetType` |
| `MARKET_STATUS` | `GenericDTO.marketStatusData` (exchange, segment, status) |
| `NSE_CD_OI` / `NSE_FO_OI` / `BSE_FO_OI` | `GenericDTO.mbpData` |
| `NSE_CM_ALL` / `NSE_CD_ALL` / `NSE_FO_ALL` / `BSE_CM` / `BSE_FO_ALL` / `MCX_PKT` | `GenericDTO.mbpData` |
| `NSE_CM_CIRC` / `NSE_CD_CIRC` / `NSE_FO_CIRC` | `GenericDTO.mbpData` |
| `NSE_FO_GREEK` / `BSE_FO_GREEK` | `GenericDTO.greekData` |
| `NSE_INDEX` / `BSE_INDEX` | `GenericDTO.indexData` |
| `ORDER` / `TRADE` | `GenericDTO.order` / `GenericDTO.trade` |

> The WebSocket uses Protobuf, not JSON. The `.proto` file is linked on the docs page and must be downloaded separately to decode `mbpData`, `greekData`, `indexData`, etc. Bookmark: [WebSocket page](https://developer.hdfcsky.com/sky-docs/docs/websocket).

---

## 5. Historical data

### 5.1 Chart candles — `GET /oapi/charts-api/charts/v1/fetch-candle`

```
GET /oapi/charts-api/charts/v1/fetch-candle
    ?api_key=<api_key>
    &symbol=RELIANCE
    &exchange=NSE
    &chartType=MINUTE     # MINUTE or DAY
    &seriesType=EQ        # derivative series type
    &start=2025-04-10
    &end=2025-04-15
Headers: Authorization: <access_token>
```

Response shape (`data.results` is an array of OHLCV tuples):
```json
{
  "data": {
    "results": [
      [4066.0, 4089.8, 4006.35, 4015.5, 1928049.0, <timestamp>]
    ]
  }
}
```

> The docs don't publish max-range-per-call or historical-depth tables (unlike Groww §5). Start with ≤7 days for minute data and iterate.

---

## 6. Security master

### `GET /oapi/v1/security-master` (download)

Returns a consolidated **CSV** catalog of all tradable securities across NSE, BSE, MCX (cash, F&O, currency, commodity). Import once at startup and cache locally — this is the equivalent of Groww's `get_all_instruments()`.

> The docs page itself only describes the CSV format at a high level; the actual endpoint path is embedded in the CSV download link. Fetch and inspect on the first integration.

---

## 7. Orders (write — NOT USED)

> All endpoints below mutate broker state. Stay in paper-trade mode unless the user explicitly approves wiring.

### Common body params (regular + AMO + BO/CO)

| Field | Type | Description |
|-------|------|-------------|
| `exchange` | string | `NSE` / `BSE` / `NFO` / `CDS` / `MCX` |
| `instrument_token` | string | Unique instrument id (from Security Master) |
| `client_id` | string | User/client id |
| `order_type` | string | `LIMIT` / `MARKET` / `SL` / `SLM` |
| `amo` | bool | `true` / `false` |
| `price` | number | Non-zero |
| `quantity` | number | Non-zero |
| `trigger_price` | number | Non-zero (for SL / SLM) |
| `disclosed_quantity` | number | Non-negative |
| `validity` | string | `DAY` or `IOC` |
| `product` | string | `CNC` / `MIS` / `NRML` |
| `order_side` | string | `BUY` or `SELL` |
| `device` | string | `Web` / `Mobile` |
| `user_order_id` | number | Client-side dedup id |
| `execution_type` | string | `REGULAR` / `BO` / `CO` / `AMO` |

### 7.1 Regular orders

| Action | Method | Path |
|--------|--------|------|
| Place | `POST` | `/oapi/v1/orders` |
| Modify | `PUT` | `/oapi/v1/orders` |
| Cancel | `DELETE` | `/oapi/v1/orders/<oms_order_id>?execution_type=REGULAR` |

### 7.2 AMO (After Market Orders)

Same endpoints as regular; set `amo=true` in the body and use `execution_type=AMO`.

| Action | Method | Path |
|--------|--------|------|
| Place | `POST` | `/oapi/v1/orders` |
| Modify | `PUT` | `/oapi/v1/orders` |
| Cancel | `DELETE` | `/oapi/v1/orders/kart/<oms_order_id>?execution_type=AMO` |

### 7.3 Conditional (Bracket + Cover)

All at `/oapi/v1/orders/kart` — differentiated by body fields (`execution_type=BO` or `CO`, plus stop-loss / target legs).

| Action | Method | Path |
|--------|--------|------|
| Place CO | `POST` | `/oapi/v1/orders/kart` |
| Place BO | `POST` | `/oapi/v1/orders/kart` |
| Modify CO | `PUT` | `/oapi/v1/orders/kart` |
| Modify BO | `PUT` | `/oapi/v1/orders/kart` |
| Exit CO | `DELETE` | `/oapi/v1/orders/kart/<oms_order_id>` |
| Exit BO | `DELETE` | `/oapi/v1/orders/kart/<oms_order_id>` |

- **BO** = Bracket Order (target + stop-loss alongside the entry).
- **CO** = Cover Order (mandatory stop-loss; buy CO's `limit > trigger`, sell CO's `limit < trigger`).

### 7.4 Order book & history

| Endpoint | Method | Path | Use |
|----------|--------|------|-----|
| Pending orders | `GET` | `/oapi/v1/orders?type=pending&client_id=...` | List open orders |
| Completed orders | `GET` | `/oapi/v1/orders?type=completed&client_id=...` | List executed orders |
| Order history | `GET` | `/oapi/v1/order/<oms_order_id>/history` | Status timeline for one order |
| Trade book | `GET` | `/oapi/v1/trades?client_id=...` | Executed trades (fills) |

Shared query params for book endpoints: `client_id`, `api_key`, optional `source`, optional `tags[]`.

---

## 8. GTT Orders — `/oapi/v1/event/gtt`

Good-Till-Triggered: single conditional order, trigger valid up to 1 year. Fires at the threshold; order is placed and attempts to fill at the limit price (subject to funds).

| Action | Method | Path |
|--------|--------|------|
| Place | `POST` | `/oapi/v1/event/gtt` |
| Modify | `PUT` | `/oapi/v1/event/gtt` |
| Cancel | `DELETE` | `/oapi/v1/event/gtt/<client_id>/<id>` |
| Fetch | `GET` | `/oapi/v1/event/gtt/<client_id>` |

Fetch returns actiontype, exchange, instrument_token, trigger price, etc.

> HDFC Sky does **not** document a separate OCO (one-cancels-other) endpoint — unlike Groww's Smart Orders. If bracket-like behavior is needed, compose it via BO (§7.3) or pair two GTTs client-side.

---

## 9. Basket Orders — `/oapi/v1/basket*`

Basket = named collection of orders submitted atomically. Two sub-types: **Normal** and **Hedge** (the Hedge variant accepts `LIMIT`/`MARKET`/`SL`/`SLM` × `CNC`/`MIS`/`NRML`).

| Action | Method | Path |
|--------|--------|------|
| Create basket | `POST` | `/oapi/v1/basket` |
| Rename basket | `PUT` | `/oapi/v1/basket` |
| Delete basket | `DELETE` | `/oapi/v1/basket` |
| Fetch baskets | `GET` | `/oapi/v1/basket/<login_id>` |
| Add instrument | `POST` | `/oapi/v1/basket/order` |
| Edit basket details | `PUT` | `/oapi/v1/basket/order` |
| Edit basket instrument | `PUT` | `/oapi/v1/basket/order` |
| Delete basket instrument | `DELETE` | `/oapi/v1/basket/order` |
| **Execute basket** | `POST` | `/oapi/v1/orders/kart` |

Constraints: instrument count per basket is limited (exact cap not published), and adding a duplicate instrument to the same basket is rejected.

---

## 10. Portfolio

| Endpoint | Method | Path | Query |
|----------|--------|------|-------|
| Holdings | `GET` | `/oapi/v1/holdings` | `client_id`, `api_key` |
| Positions (Daywise) | `GET` | `/oapi/v1/positions` | `client_id`, `api_key`, `type=live` |
| Positions (Netwise) | `GET` | `/oapi/v1/positions` | `client_id`, `api_key`, `type=historical` |
| Convert position | `PUT` | `/oapi/v1/position/convert` | CNC↔MIS with partial quantity supported |
| eDIS authorize holdings | `POST` | `/oapi/v1/edis/instrument_details` | Authorize debit for a sell |

Holdings response fields (observed): `branch_code`, `buy_avg`, `buy_avg_mtm`, `client_id`, `exchange`, `free_quantity`, `instrument_details{exchange, instrument_name, instrument_token}`, ...

Positions response fields: `average_buy_price`, `average_sell_price`, `buy_amount`, `buy_quantity`, `cf_buy_*` (carry-forward), `client_id`, ... mirroring Groww's positions API.

---

## 11. Funds & Margin

### 11.1 Funds V1 — `GET /oapi/v1/funds/view`

Legacy endpoint returning a list of categories (`id`, `title`). Query: `client_id`, `type=all`, `api_key`. Prefer V2.

### 11.2 Funds V2 — `GET /oapi/v2/funds/view`

Tabular fund details: `Available Margin`, `Margin Used`, `Opening Balance`, `Adhoc Deposit`, `Notion`, `Pay In`, `Pay out`, `Pledge Benefit`, `Equity Credit Sell`, `Option Credit For Sell`, ...

Response shape:
```json
{
  "data": {
    "client_id": "DEMO1",
    "headers": ["Description", ""],
    "values": [
      { "0": "Available Margin", "1": "8239.85" },
      { "0": "Margin Used",      "1": "19.34"   }
    ]
  }
}
```

### 11.3 Margin calculator — `POST /oapi/v1/margin`

Pre-check margin for a hypothetical order (equivalent of Groww's `get_order_margin_details`).

Body:
```json
{
  "data": [{
    "segment": "FutOpt",
    "series":  "OPTIDX",
    "exchange": "BFO",
    "side":    "SELL",
    "mode":    "NEW",
    "symbol":  "SENSEX24AUG24200CE",
    "underlying": "76000",
    "token":   "74643",
    "quantity":"25",
    "price":   "332.85",
    "product": "0"
  }]
}
```

Response includes `combined_margin.delivery_margin`, `span`, `somtier_margin`, ...

### 11.4 Brokerage calculator — `GET /oapi/v1/brokerage/calculate`

Compute brokerage + charges for a hypothetical order via query params (`exchange`, `symbol`, `quantity`, `buy_price`, `sell_price`, `segment`, `product`, `api_key`).

### 11.5 P&L report — `POST /oapi/v1/reports/pnl`

```
POST /oapi/v1/reports/pnl?api_key=<api_key>
Body: {
  "tradingAccountDetails": { "accountSettlementType": 0 },
  "portfolioProfitLossDetails": {
    "portfolioName": "DEFAULT",
    "userOwner": 1,
    "assetClass": 99,
    "product": 99,
    "reportType": 1,
    "chargesType": "N",
    "initialType": "Y",
    "fromDate": "<from>",
    "toDate":   "<to>"
  },
  "send_mail": true
}
```

---

## 12. Profile

### `GET /oapi/v1/user/trading_info?client_id=...&api_key=...`

Returns user identity + enabled segments:

```json
{
  "data": {
    "client_id": "TEST1",
    "name": "DEMO",
    "ddpi": false,
    "poa_enabled": true,
    "exchanges_subscribed": ["NFO", "NSE", "BSE", "CDS"],
    "products_enabled":     ["NRML", "CNC", ...]
  }
}
```

> `ddpi` flag = Demat Debit and Pledge Instruction (same concept as Groww's `ddpi_enabled` in `get_user_profile`).

---

## 13. Response & error structure

### 13.1 Standard response envelope

Most endpoints return:
```json
{ "data": { ... }, "meta": { "statusCode": "OK", "statusMsg": "OK" } }
```
or
```json
{ "data": { ... }, "message": "", "status": "success" }
```

### 13.2 HTTP status codes

| Code | Meaning |
|------|---------|
| **200** | Success |
| **400** | Bad Request — invalid/malformed body |
| **401** | Unauthorized — missing/invalid token (re-auth) |
| **403** | Forbidden — valid token but operation not permitted (often missing `User-Agent` or disabled segment) |
| **404** | Not Found — resource doesn't exist |
| **422** | Unprocessable — semantically invalid request |
| **5xx** | Server error — retry with backoff |

### 13.3 Rejection codes

Internal rejection codes are distributed as a downloadable CSV on the [Rejection Code](https://developer.hdfcsky.com/sky-docs/docs/rejection_code) page. Fetch it once and index by code; display `message` to the user when an order is rejected.

---

## 14. How our agent uses HDFC Sky today

| Area | HDFC Sky surface | Our code | Notes |
|------|------------------|----------|-------|
| Auth | 6-step REST flow | **Not wired.** | User has creds (see user_profile memory) but no client exists. |
| Live price | `PUT /fetch-ltp`, WebSocket | **Not wired.** | Groww LTP is authoritative. |
| Historical | `GET /charts-api/.../fetch-candle` | **Not wired.** | `data_fetcher` uses Groww + yfinance. |
| Holdings / positions | `/holdings`, `/positions` | **Not wired.** | Groww MCP handles the safety check. |
| Funds / margin | `/v2/funds/view`, `/margin` | **Not wired.** | — |
| Orders (all) | `/orders`, `/orders/kart`, `/event/gtt`, `/basket*` | **Not wired — paper-trade only.** | — |

### When would we wire this?

- **Quote redundancy:** if Groww throttles or goes down, HDFC Sky's `/fetch-ltp` + WebSocket could serve as a fallback for the watchlist loop.
- **Holdings reconciliation:** if we start routing some orders through HDFC Sky (better intraday margin, different brokerage), we need a holdings/positions read path.
- **Different product enablement:** HDFC Sky supports MCX commodities and BSE FO natively (CDS subscription visible in `products_enabled`); Groww's MCX was only added in SDK 1.5.

Integration shape if/when wired:
1. Add `hdfc_sky_client.py` alongside `groww_client.py` with `_generate_access_token`, `fetch_live_price`, `fetch_candles`, `fetch_holdings`, `fetch_positions`.
2. Token refresh is harder than Groww — the 6-step flow is interactive (OTP + PIN). Options:
   - Maintain a long-lived `accessToken` refreshed manually (daily manual OTP acceptable for R&D).
   - Integrate HDFC's backend-only flow with an OTP delivery channel you control (SMS gateway webhook) — non-trivial.
3. Reuse the `_live_data_limiter` pattern (sliding-window 5 rps initially) since HDFC doesn't publish REST limits.

---

## 15. Source URLs (for re-sync)

- Index: https://developer.hdfcsky.com/sky-docs/docs/intro
- **Auth flow:**
  - Token ID: https://developer.hdfcsky.com/sky-docs/docs/fetch_access_token_via_api/token_id
  - Validate username: https://developer.hdfcsky.com/sky-docs/docs/fetch_access_token_via_api/validate
  - Validate OTP: https://developer.hdfcsky.com/sky-docs/docs/fetch_access_token_via_api/OTP
  - Validate PIN: https://developer.hdfcsky.com/sky-docs/docs/fetch_access_token_via_api/2FA
  - Resend OTP: https://developer.hdfcsky.com/sky-docs/docs/fetch_access_token_via_api/resend2fa
  - Authorize: https://developer.hdfcsky.com/sky-docs/docs/fetch_access_token_via_api/authorize
  - Access token: https://developer.hdfcsky.com/sky-docs/docs/fetch_access_token_via_api/access_token
  - Frontend flow: https://developer.hdfcsky.com/sky-docs/docs/token_id_fe
- **Market data:**
  - LTP: https://developer.hdfcsky.com/sky-docs/docs/fetchltp
  - Candles: https://developer.hdfcsky.com/sky-docs/docs/fetchCandles
  - WebSocket: https://developer.hdfcsky.com/sky-docs/docs/websocket
  - Security master: https://developer.hdfcsky.com/sky-docs/docs/security_master
- **Orders:**
  - Regular place/modify/delete: https://developer.hdfcsky.com/sky-docs/docs/orders/regular_order/placeregularorder (+ sibling pages)
  - AMO place/modify/delete: https://developer.hdfcsky.com/sky-docs/docs/orders/regular_order/placeamoorder
  - BO/CO: https://developer.hdfcsky.com/sky-docs/docs/orders/conditional_orders/placeboorder (+ siblings)
  - Order book: https://developer.hdfcsky.com/sky-docs/docs/orders/order_book/fetchpendingorder (+ siblings)
  - GTT: https://developer.hdfcsky.com/sky-docs/docs/gttorders/placegttorder (+ siblings)
  - Basket: https://developer.hdfcsky.com/sky-docs/docs/basketorders/createbasket (+ siblings)
- **Portfolio / funds / margin:**
  - Holdings: https://developer.hdfcsky.com/sky-docs/docs/portfolio/fetchdematholdings
  - Positions: https://developer.hdfcsky.com/sky-docs/docs/portfolio/ (daywise) & /portfolio/fetchpositionsnetwise
  - Convert position: https://developer.hdfcsky.com/sky-docs/docs/portfolio/convertpositions
  - eDIS: https://developer.hdfcsky.com/sky-docs/docs/portfolio/edis
  - Funds V1: https://developer.hdfcsky.com/sky-docs/docs/funds/
  - Funds V2: https://developer.hdfcsky.com/sky-docs/docs/funds/fetchfundsv2
  - Margin: https://developer.hdfcsky.com/sky-docs/docs/margin_calculator
  - Brokerage: https://developer.hdfcsky.com/sky-docs/docs/brokerage/calculateBrokerage
  - P&L: https://developer.hdfcsky.com/sky-docs/docs/PNL_Data
- **Reference:**
  - Profile: https://developer.hdfcsky.com/sky-docs/docs/profile/
  - Response structure: https://developer.hdfcsky.com/sky-docs/docs/response
  - Error structure: https://developer.hdfcsky.com/sky-docs/docs/error_structure
  - Rejection code CSV: https://developer.hdfcsky.com/sky-docs/docs/rejection_code

### Re-sync procedure (next agent)

The docs site is a **client-rendered Docusaurus 3.x SPA** — `curl` of the page URL returns only a ~4KB shell. To re-sync:

1. Fetch `https://developer.hdfcsky.com/sky-docs/sitemap.xml` for the full URL list.
2. Grab `https://developer.hdfcsky.com/sky-docs/assets/js/runtime~main.<hash>.js` — contains two chunkId→hash maps.
3. Grab `https://developer.hdfcsky.com/sky-docs/assets/js/main.<hash>.js` — contains route→content-hash map (grep for `component:d\("/sky-docs/docs/...","<id>"\)` and the paired `"<id>":\{"__comp":...,"content":"<hash>"\}`).
4. Fetch each content chunk at `/sky-docs/assets/js/<contentHash>.<fileHash>.js` and run it through the `extract_hdfc.py` parser (checked out-of-tree — scratch script, was removed after use).

If HDFC switches to SSG builds, this is unnecessary — `curl` of each page URL will return readable HTML.
