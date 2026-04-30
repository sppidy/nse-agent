# nse-agent

Paper-trading agent for NSE (Indian equities). Part of [`trading-agent`](https://github.com/sppidy/trading-agent).

- Rule-based strategy (RSI / EMA crossovers) + Gemini/Groq/Cloudflare LLM signals (cascade fallback)
- Local CatBoost direction predictor (~45 engineered features) with walk-forward retraining and promotion gates
- News-sentiment enrichment with sanitised prompts
- Autopilot loop with exponential backoff and circuit breaker per LLM provider
- Dual paper portfolios in `config.py` — `main` (Rs.1cr, ML data harvesting) and `eval` (Rs.10k, realistic small-capital evaluation)
- **Data source: yfinance by default; Groww optional.** Groww live-data API is IP-whitelisted (server-only), so the agent transparently falls back to yfinance for OHLCV, live quotes, and fundamentals if `GROWW_API_KEY` isn't set or the call fails — no configuration needed for casual users.

> **Disclaimer:** Paper-trading only. No live order code. Not financial advice.

## Quick start

```bash
python -m venv venv && source venv/bin/activate          # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
cp .env.example .env                                      # then fill in at least one LLM key
python main.py help
python main.py status
python main.py autopilot --force                          # market-hours simulation
```

## CLI commands

| Command | What it does |
| --- | --- |
| `scan`       | Rule-based watchlist scan |
| `ai-scan`    | LLM watchlist scan |
| `backtest`   | Per-symbol + portfolio-level walk-forward backtest |
| `train`      | Train CatBoost predictor with walk-forward metrics |
| `predict`    | Next-session directional predictions |
| `trade` / `ai-trade` | Single trading cycle |
| `autopilot`  | Continuous loop during market hours |
| `chat`       | Interactive terminal assistant |

## Reliability

- Portfolio/journal/training-log writes are atomic, file-locked.
- AI output is normalised to a strict BUY/SELL/HOLD + confidence schema before any trade decision.
- ML retraining is cadence-gated; the new model is only promoted if it beats the incumbent on validation.
- Model files are integrity-checked via SHA-256 sidecar (`predictor.pkl.sha256`).

## Scheduling

- Linux/macOS: `setup_cron.sh` installs Mon-Fri 9 AM IST start / 3:45 PM IST stop.
- Windows: `setup_scheduler.ps1` registers a Task Scheduler entry resolved from the script's own path.

## License

[Apache-2.0](LICENSE). Contributing guidelines and security policy live in the [super-repo](https://github.com/sppidy/trading-agent).
