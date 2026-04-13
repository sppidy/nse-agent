# AI Trading Agent


Paper-trading assistant for NSE stocks with:
- rule-based strategy (RSI/EMA),
- Gemini-assisted signal generation,
- news sentiment enrichment,
- local ML direction predictor,
- autopilot execution loop.

## Quick start

1. Create and activate virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Copy env template:
   - `copy .env.example .env` (Windows)
   - `cp .env.example .env` (Linux/macOS)
4. Add your Gemini API key in `.env`.
5. Run:
   - `python main.py help`
   - `python main.py status`
   - `python main.py autopilot --force` (test mode)

## Main commands

- `python main.py scan` - rule-based watchlist scan
- `python main.py ai-scan` - AI-based scan
- `python main.py backtest` - per-symbol + portfolio-level backtest
- `python main.py train` - train local predictor (with walk-forward metrics)
- `python main.py predict` - next-session directional predictions
- `python main.py trade` / `ai-trade` - single trading cycle
- `python main.py autopilot` - continuous loop during market hours
- `python main.py chat` - interactive terminal assistant

## Reliability and data safety

- Portfolio/journal/training log writes are atomic.
- JSON reads/writes are file-locked to reduce race risk.
- Autopilot loop now survives transient runtime failures with backoff.
- AI output is normalized to strict signal schema before trade decisions.
- ML retraining is cadence-gated and includes model-promotion checks (new model is kept only if quality is stable/improved).

## Security notes

- Never commit `.env`.
- If an API key is exposed, rotate it immediately in your provider console.
- News headlines are sanitized before being injected into AI prompts.
- Model files are integrity-checked via SHA-256 sidecar (`predictor.pkl.sha256`).

## Windows scheduling

- `start_autopilot.bat` now uses script-local paths and `PYTHON_EXE` override.
- `setup_scheduler.ps1` resolves paths from script location (no hardcoded repo path).

## Troubleshooting

- Missing modules (`rich`, `prompt_toolkit`): reinstall `pip install -r requirements.txt`.
- Model integrity error: run `python main.py train` to regenerate model + hash.
- No prices/data: verify market availability and internet connectivity.
