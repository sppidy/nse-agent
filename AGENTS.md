# AGENTS.md

Orientation for AI agents working in this ecosystem. This file documents the full multi-repo system so an agent landing in any one of these directories can understand the whole.

## The five repos

This project spans **five related repositories** on the user's machine (Windows, paths use `B:\`). Only two are git-tracked; the others are deployed by scp.

| # | Repo | Path | Role | Git | Deploy |
|---|------|------|------|-----|--------|
| 1 | NSE agent | `B:\projects\ai-trading-agent` | Python trading agent for Indian equities | ✅ | `git pull` on server |
| 2 | NSE backend | `B:\projects\ai-agent-trading-app\backend` | FastAPI wrapper around NSE agent | ✅ | `scp` to `~/backend/` |
| 3 | Forex agent | `B:\projects\forex-trading-agent\agent` | Python trading agent for FX/commodities | ❌ | `scp` (whole dir) |
| 4 | Forex backend | `B:\projects\forex-trading-agent\backend` | FastAPI multi-user wrapper around forex agent | ❌ | `scp` (whole dir) |
| 5 | Android app | `B:\projects\ai-agent-trading-app\android-app` | Kotlin/Compose mobile frontend | ✅ | APK build |

Server: `ubuntu@BACKEND_HOST` (self-hosted).

## Deployment rules — read before deploying

- **NSE agent:** `git pull` only. Never `scp` the full repo. It clobbers live state files (`portfolio.json`, `*.db`, `trade_journal.json`) on the server. A past incident wiped live portfolio data this way.
- **NSE backend:** `scp -r` is fine. Runs under **systemd** as `ai-trader-api.service` and `ai-trading-agent.service`. Restart with `sudo systemctl restart ai-trader-api.service ai-trading-agent.service`.
- **Forex agent + backend:** `scp` both directories. No git repos locally. Runs under **Docker** (not systemd) — restart with `docker compose restart` or equivalent. Safe to scp because forex state lives in PostgreSQL, not files.

## How they fit together

```
Android app ──┬──► NSE backend (FastAPI, single-user, in-memory jobs)
              │       └── dynamically imports NSE agent modules via AGENT_DIR
              │
              └──► Forex backend (FastAPI, multi-user, PostgreSQL, Docker)
                      └── imports Forex agent modules
```

The Android app has a configurable base URL — one app talks to either backend.

## NSE agent (repo #1)

**Stack:** Python, CatBoost, scikit-learn, Gemini API, yfinance, feedparser.

**Entry point:** `main.py` — CLI with commands:
- `autopilot` — market-hours loop with backoff
- `scan` / `ai-scan` — rule-based or AI watchlist scan
- `trade` / `ai-trade` — single trading cycle
- `train` — walk-forward CatBoost retraining with promotion gates
- `backtest` — per-symbol and portfolio backtests
- `chat` — interactive terminal assistant

**Key modules:**
- `predictor.py` — CatBoost model (~45 features: RSI, EMA, MACD, Bollinger, ATR, Stochastic, ADX, volatility, volume, S/R). Predicts ≥2% upside within 5 trading days. Model files in `models/` with SHA-256 integrity.
- `ai_strategy.py` — Gemini-powered signals, strict BUY/SELL/HOLD + confidence schema.
- `paper_trader.py` — Portfolio mgmt, SL/TP, atomic journal writes.
- `autopilot.py` — Market-hours loop.
- `strategy.py` — Rule-based (RSI + EMA crossovers).
- `news_sentiment.py` — Headline scraping and sanitization.
- `data_fetcher.py` — yfinance with exponential backoff.
- `config.py` — Watchlist (NSE tickers), `INITIAL_CAPITAL`, position sizing, SL/TP defaults.

**State files (DO NOT overwrite via scp):** `portfolio.json`, `trade_journal.json`, `*.db`.

## NSE backend (repo #2)

**Stack:** FastAPI, uvicorn, in-memory job store (TTL 600s, max 200 jobs).

**Entry point:** `api_server.py` (port 8000).

**How it loads the agent:** Dynamically imports NSE agent from `../ai-trading-agent/` or the `AGENT_DIR` env var. So **changes to NSE agent require the backend service to be restarted**.

**Key behaviors:**
- Bearer auth via `API_AUTH_TOKEN`.
- CORS-configurable.
- Per-endpoint rate limiting.
- Thread-safe portfolio mutex (`_PORTFOLIO_LOCK`).
- WebSocket log streaming at `/logs/stream` via an `AsyncQueueHandler`.
- Autopilot control wraps the systemd `ai-trading-agent.service`.

**Endpoints:** `/api/status`, `/api/prices`, `/api/scan`, `/api/ai-scan`, `/api/trade`, `/api/ai-signals/apply`, `/api/autopilot/{start,stop}`, `/api/chat`, `/api/journal`, `/api/logs/{dates,recent}`, `/api/training-log`, `/logs/stream` (WS).

## Forex agent (repo #3)

**Stack:** Python, shares base infrastructure with NSE agent (`data_fetcher`, `backtester`, `paper_trader`, `learner`).

**Forex-specific modules:**
- `forex_strategy.py` — **ICT Asian Sweep**: Asian session (7 PM–12 AM ET) high/low sweep + CISD detection + Fib 50% retracement entry.
- `london_breakout.py` — Pre-London range breakout (3 AM–12 PM ET).
- `killzone_reversal.py` — ICT kill zone reversals (London/NY opens).
- `strategy_engine.py` — Maps pairs to strategies with trading windows (IST/ET/UTC).

**Config:**
- `INITIAL_CAPITAL` = $100,000
- `MAX_POSITION_SIZE_PCT` = 5% (tighter than NSE's 10%)
- Data interval: 15-minute candles (NSE uses daily)
- Watchlist: FX majors (EURUSD, GBPUSD, USDJPY, AUDUSD, …) + gold/silver (GC=F, SI=F), USD Index (DX-Y.NYB) for regime.

## Forex backend (repo #4)

**Stack:** FastAPI + **PostgreSQL** + Docker. Multi-user.

**Entry point:** `api_server.py` (port 8000 via Docker).

**Auth:** `X-API-Key` header. Admin vs user roles.

**Schema (`users.py`):** users, portfolios, positions, orders, trades, audit logs — all user-scoped.

**Extra endpoints vs NSE backend:** `/api/admin/users`, `/api/strategies`, `/api/market-regime`, `/api/candles`.

## Android app (repo #5)

**Stack:** Kotlin, Jetpack Compose, Retrofit2, Room (local cache), WebSocket, biometric auth.

**Architecture:**
- `ApiService.kt` — Retrofit REST + WebSocket client.
- `TradingRepository.kt` — Repo pattern over API + cache.
- `LogWebSocket.kt` — Live log streaming.
- `AppPreferences.kt` — Encrypted storage for base URL + API key.
- `BiometricAuth.kt` — Fingerprint/face unlock.
- `BackgroundMonitorService.kt` — Portfolio monitor when app is backgrounded.

**Screens:** Dashboard, Chart, Scan, Chat, Log, Settings.

**Key fact:** configurable base URL — same APK talks to NSE backend or Forex backend.

## Shared infrastructure between agents

Both agents share: `data_fetcher`, `backtester`, `paper_trader`, `learner`, `market_calendar` (IST/ET/UTC logic). Changes here affect both markets — verify with backtests in both.

## When editing

- When user says "the agent" without context, **ask NSE or Forex**.
- Changing NSE agent code → redeploy NSE backend service too (it imports agent modules).
- Changing Android API contracts → keep both backends in sync.
- Capital assumptions for the primary user are small (~Rs.1000). Don't suggest strategies that require large capital.
- User is new to algo trading and has Groww + HDFC Sky broker accounts (NSE side).

## External services

- **yfinance** — OHLCV data for both markets.
- **LLM cascade** (see `ai_strategy.py` in each agent + chat path in each backend): Copilot/Haiku → **Ollama (`gemma4:e4b` @ `http://BACKEND_HOST:11434`, self-hosted on self-hosted)** → OpenRouter → Groq → Cloudflare → Gemini. Override Ollama with env vars `OLLAMA_BASE_URL` / `OLLAMA_MODEL`.
- **Kaggle API** — Remote model training (NSE CatBoost retraining).
- **Brokers** (expected integrations, not fully wired yet): Groww, HDFC Sky for NSE; TBD for Forex.
