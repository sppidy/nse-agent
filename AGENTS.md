# AGENTS.md

Orientation for AI agents working in this ecosystem. This file documents the full multi-repo system so an agent landing in any one of these directories can understand the whole.

## The codebases

Code is organised across **two Windows top-level projects** (`B:\projects\ai-trading-agent` and `B:\projects\ai-agent-trading-app`) plus the Forex project. Several sub-folders of `ai-agent-trading-app` are deployed independently.

| # | Module | Path | Role | Git | Deploy |
|---|--------|------|------|-----|--------|
| 1 | NSE agent | `B:\projects\ai-trading-agent` | Python trading agent for Indian equities | ✅ `sppidy/ai-trading-agent` | `git pull` on server |
| 2 | NSE backend | `B:\projects\ai-agent-trading-app\backend` | FastAPI wrapper around NSE agent | ✅ `sppidy/ai-trading-agent-utils` | `scp` to `~/backend/` |
| 3 | Web dashboard | `B:\projects\ai-agent-trading-app\frontend` | Vanilla JS SPA served by the backend | ✅ (same repo) | `scp` to `~/frontend/` |
| 4 | Desktop app | `B:\projects\ai-agent-trading-app\desktop-app` | Native WinUI 3 (.NET 8) client | ✅ (same repo) | local build |
| 5 | Android app | `B:\projects\ai-agent-trading-app\android-app` | Kotlin / Jetpack Compose mobile client | ✅ (same repo) | APK build, side-load |
| 6 | Forex agent | `B:\projects\forex-trading-agent\agent` | Python trading agent for FX/commodities | ❌ | `scp` (whole dir) |
| 7 | Forex backend | `B:\projects\forex-trading-agent\backend` | FastAPI multi-user wrapper around forex agent | ❌ | `scp` (whole dir) |

Server: `ubuntu@${BACKEND_HOST}`. All NSE-side services are private-network-only.

## Deployment rules — read before deploying

- **NSE agent:** `git pull` only. Never `scp` the full repo. It clobbers live state files (`portfolio.json`, `*.db`, `trade_journal.json`) on the server. A past incident wiped live portfolio data this way.
- **NSE backend:** `scp` specific files into `~/backend/` (preserve server-only files like `certs/` and `.backup`). Runs under **systemd** as `ai-trader-api.service` and `ai-trading-agent.service`. Restart with `sudo systemctl restart ai-trader-api.service ai-trading-agent.service`.
- **Web dashboard:** `scp` the three files (`index.html`, `styles.css`, `app.js`) + the vendored `lightweight-charts.js` into `~/frontend/`. The backend serves them statically from `/dashboard`; no service restart needed for HTML/CSS/JS updates.
- **Desktop app:** built locally via `dotnet build -p:Platform=arm64 -c Debug` — no server deploy. Runs from `desktop-app/NEON.Trader.Desktop/bin/arm64/Debug/net8.0-windows10.0.19041.0/NEON.Trader.exe`. Uses Windows App SDK 2.0-preview2 + LiveCharts2 + JetBrains Mono bundled font.
- **Android app:** `./gradlew assembleRelease` with keystore creds from `.secrets/keystore.env` produces `app-release.apk` (signed, v2 scheme). Copy to repo root as `AITrader-release.apk` and side-load with `adb install -r`.
- **Forex agent + backend:** `scp` both directories. No git repos locally. Runs under **Docker** (not systemd) — restart with `docker compose restart` or equivalent. Safe to scp because forex state lives in PostgreSQL, not files.

## How they fit together

```
Web dashboard ─┐
Desktop app ───┤     ┌── NSE backend (FastAPI, single-user, in-memory jobs)
Android app  ──┼────►│      └── dynamically imports NSE agent modules via AGENT_DIR
               │     │
               └────►└── Forex backend (FastAPI, multi-user, PostgreSQL, Docker)
                              └── imports Forex agent modules
```

**All three clients (web, desktop, Android) share the same HTTP + WebSocket contract.** Each stores a configurable base URL + API key per "profile", so one client can flip between NSE main, NSE eval, and Forex backends without reinstall.

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
- `ai_strategy.py` — Gemini-powered signals, strict BUY/SELL/HOLD + confidence schema. News-sentiment fetch has a **45 s** timeout (bumped from 15 s) because the LLM sentiment pass is load-sensitive and used to drop to NEUTRAL too often.
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
- Auth via `X-API-Key` header (value from `API_AUTH_TOKEN`).
- CORS-configurable.
- Per-endpoint rate limiting (distinct buckets: `scan:start`, `scan:status`, `trade`, `chat:start`, `chat:status`).
- Thread-safe portfolio mutex (`_PORTFOLIO_LOCK`) — serialises `/api/trade`, `/api/ai-signals/apply`, and `/api/order` against each other and against the autopilot.
- **LogBroadcaster** does double duty: it hooks the in-process Python `logger` **and** tails `logs/trading_agent.log` on disk, so the separate `ai-trading-agent.service` (autopilot) process — which only writes to the file — also streams through the WebSocket. Starts from EOF so new clients don't get a history dump.
- `TIMEFRAME_TO_YF` lookup is case-insensitive: `5m`, `15m`, `1h`, `1d`, `1w`, `1mo`, `1y` all resolve.
- Autopilot control wraps the systemd `ai-trading-agent.service`.
- Static-files mount at `/dashboard` serves the web frontend (`../frontend/`).

**Endpoints (all prefixed `/api/`):**
- Read: `status`, `prices`, `candles`, `market-regime`, `watchlist`, `journal`, `lessons`, `logs/dates`, `logs/recent`, `training-log`.
- Scan: `scan` (sync rule-based), `ai-scan` (async job), `scan/status/{job_id}`.
- Trade: `trade` (rule-based cycle), `ai-signals/apply` (apply pre-computed signals), **`order`** (manual BUY/SELL from client UIs — required for desktop + Android Portfolio pages).
- Autopilot: `autopilot/start`, `autopilot/stop`.
- Chat: `chat`, `chat/status/{job_id}`.
- Plus WebSocket: **`/ws/logs`** (token via `X-API-Key` header or `?token=` query).

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

## Web dashboard (module #3)

**Stack:** vanilla JS (no build), HTML + CSS, TradingView `lightweight-charts` vendored locally, JetBrains Mono from Google Fonts.

**Files:** `index.html`, `styles.css`, `app.js`, `lightweight-charts.js`.

**Served by:** NSE backend at `https://<host>/dashboard` (backend static-files mount).

**Views:** Dashboard, Watchlist, Charts (candle + indicator toggles + RSI sub-pane), Strategy (rule builder + client-side backtester + AI-generate via `/api/chat` with a JSON-only system prompt), Scanner, Agent chat, Logs (WebSocket live stream), Settings.

**NEON visual language:** dark grid (`#05060a`) + neon-lime accent (`#b4ff00`), scanline overlay, ticker strip, cascadia-mono / jetbrains-mono monospace. Same palette is mirrored in the desktop + Android apps.

## Desktop app (module #4)

**Stack:** WinUI 3 + .NET 8 (`net8.0-windows10.0.19041.0`), Windows App SDK 2.0-preview2, CommunityToolkit.Mvvm source generators, LiveChartsCore.SkiaSharpView.WinUI 2.0-rc6, SkiaSharp.Views.WinUI 3.119, bundled JetBrains Mono TTF. Platforms: x86 / x64 / arm64.

**Project:** `desktop-app/NEON.Trader.Desktop.sln`. Unpackaged (`WindowsPackageType=None`), runs as a plain `.exe`.

**Architecture:**
- `Services/ApiClient.cs` — typed `HttpClient`, self-signed cert bypass (for self-hosted backend), `X-API-Key` header, every endpoint the backend exposes.
- `Services/SettingsService.cs` — multi-profile JSON persistence under `%LocalAppData%\NEON.Trader\settings.json`.
- `Services/Indicators.cs` + `Services/Backtester.cs` — pure-C# SMA/EMA/RSI/Bollinger/ATR + an intrabar SL/TP backtest engine, **byte-for-byte matching the web JS implementation**, so results agree.
- `Models/BackendProfile.cs` — NSE main / NSE eval / Forex profiles with per-profile URL + API key.
- `App.xaml.cs` — global crash logger to `%LocalAppData%\NEON.Trader\crash.log` (hooks `Application.UnhandledException`, `AppDomain.UnhandledException`, `TaskScheduler.UnobservedTaskException`).

**Views (under `Views/`):** Dashboard, Watchlist, Portfolio, Charts, Strategy, Scanner, Agent, Logs, Settings.

**Key UX details:**
- Extended title-bar (`ExtendsContentIntoTitleBar=true`) with `SetTitleBar(AppTitleBar)`. System min/max/close buttons themed to the neon palette via `AppWindow.TitleBar` colours.
- `NavigationCacheMode="Required"` on every page — switching menus preserves scroll position, chart zoom, chat history, order form text.
- Charts pane uses **bar-index** X axis (`FinancialPointI` + `ObservablePoint` overlays) with a labeler that maps index → `candles[i].Time`, so overnight/weekend gaps collapse — same behaviour as TradingView. Zoom + pan synced between price and RSI sub-pane via mirrored `MinLimit` / `MaxLimit`.
- Backend timestamps are tz-naive server-local (IST) — **parsed with `DateTimeStyles.None`** everywhere (ChartsPage, Backtester, Trade.FormattedTime) so no phantom UTC↔IST shift.
- Portfolio page drives `/api/order` for manual BUY / SELL — qty empty → Kelly, price empty → live quote, per-position row has inline SELL-qty + SELL + SELL ALL.

## Android app (module #5)

**Stack:** Kotlin, Jetpack Compose, Retrofit2, Room (local cache), Navigation3, biometric auth.

**Architecture:**
- `ApiService.kt` — Retrofit REST client (all backend endpoints including `/api/order`).
- `LogWebSocket.kt` — Live log streaming via `/ws/logs`.
- `TradingRepository.kt` — Repo pattern over API + cache; `placeOrder()` auto-fills the active NSE portfolio.
- `AppPreferences.kt` — Encrypted storage for base URL + API key + active market mode (`NSE`/`FOREX`) + selected NSE portfolio (`main`/`eval`).
- `BiometricAuth.kt` — Fingerprint / face unlock gating the whole app and the Settings tab independently.
- `BackgroundMonitorService.kt` — Portfolio monitor when app is backgrounded.

**Screens (bottom nav):** Dashboard, **Portfolio** (manual BUY / SELL), Scanner, Charts, Agent chat, Logs, Settings.

**Release signing:** keystore `.secrets/aitrader-release.jks`, alias `aitrader`, passwords from `.secrets/keystore.env`, build with `./gradlew assembleRelease -PRELEASE_STORE_FILE=… -PRELEASE_STORE_PASSWORD=… -PRELEASE_KEY_ALIAS=… -PRELEASE_KEY_PASSWORD=…` — produces v2-signed APK.

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

- **Groww** — primary NSE data source. Live LTP + historical candles via REST (`groww_client.py`); market movers + fundamentals via Groww MCP at `https://mcp.groww.in/mcp/` (`groww_mcp.py`). Auth via `GROWW_API_KEY` (TOTP JWT) + `GROWW_TOTP_SECRET` (base32 seed). **No order code is wired — agent is paper-trade only.** Free tier caps all Live-Data endpoints (LTP/OHLC/Quote/historical) at 10 rps / 300 rpm shared; `groww_client._api_get` enforces 8 rps / 250 rpm client-side so we never trip 429. Full SDK reference at [`docs/GROWW_SDK.md`](docs/GROWW_SDK.md) — including a rate-limits section.
- **yfinance** — fallback for index data (Groww doesn't serve INDEX segment) and supplementary Yahoo news + options-chain PCR in `news_sentiment.py`.
- **pandas_market_calendars (XNSE)** — deterministic trading-day/holiday lookup in `market_calendar.py`. No API call.
- **LLM cascade** (see `ai_strategy.py` in each agent + chat path in each backend): Copilot/Haiku → **Ollama (`nemotron-3-nano:4b`, self-hosted, set `OLLAMA_BASE_URL`)** → OpenRouter → Groq → Cloudflare → Gemini. Ollama is chat-only (skipped on `want_json=True` — CPU inference is too slow for batch scans). Override with env vars `OLLAMA_BASE_URL` / `OLLAMA_MODEL`.
- **Kaggle API** — Remote model training (NSE CatBoost retraining).
- **Brokers** (expected integrations, not fully wired yet): Groww (read-only live data + MCP), HDFC Sky for NSE; TBD for Forex.
