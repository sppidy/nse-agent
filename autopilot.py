"""Autopilot mode — runs the AI trading bot continuously during market hours."""

import os
import time
import sys
from datetime import datetime, timedelta

import pandas as pd

import config
from data_fetcher import get_watchlist_prices, get_historical_data, get_market_regime
from paper_trader import PaperTrader, D
from ai_strategy import analyze_watchlist, get_portfolio_advice
from strategy import get_latest_signal, get_scored_signal
from learner import log_trade, record_outcome, get_snapshot, generate_lessons, print_performance_report
from predictor import predict, train_model, should_retrain
from logger import logger
from market_calendar import now_ist, is_market_trading_day, MARKET_OPEN_TIME, MARKET_CLOSE_TIME
from persistence import read_json, write_json_atomic


def is_market_open() -> bool:
    """Check if Indian stock market is currently open."""
    t = now_ist()
    # Weekday check (Mon=0, Fri=4)
    if t.weekday() > 4:
        return False
    if not is_market_trading_day(t.date()):
        return False
    market_open = t.replace(
        hour=MARKET_OPEN_TIME.hour,
        minute=MARKET_OPEN_TIME.minute,
        second=0,
        microsecond=0,
    )
    market_close = t.replace(
        hour=MARKET_CLOSE_TIME.hour,
        minute=MARKET_CLOSE_TIME.minute,
        second=0,
        microsecond=0,
    )
    return market_open <= t <= market_close


def time_to_market_open() -> timedelta | None:
    """Get time remaining until market opens. None if open."""
    if is_market_open():
        return None
    t = now_ist()
    today_open = t.replace(
        hour=MARKET_OPEN_TIME.hour,
        minute=MARKET_OPEN_TIME.minute,
        second=0,
        microsecond=0,
    )
    if t < today_open and is_market_trading_day(t.date()):
        return today_open - t
    # Next verified market session day
    days_ahead = 1
    while True:
        next_day = t + timedelta(days=days_ahead)
        if is_market_trading_day(next_day.date()):
            next_open = next_day.replace(
                hour=MARKET_OPEN_TIME.hour,
                minute=MARKET_OPEN_TIME.minute,
                second=0,
                microsecond=0,
            )
            return next_open - t
        days_ahead += 1


# Candidate pool for trending stock scanner
SCAN_POOL = [
    "TATAPOWER.NS", "BPCL.NS", "GAIL.NS", "COALINDIA.NS", "NTPC.NS",
    "BANKBARODA.NS", "CANBK.NS", "INDIANB.NS", "UNIONBANK.NS",
    "RECLTD.NS", "PFC.NS", "BHEL.NS", "BEL.NS",
    "IRCTC.NS", "RVNL.NS", "SUZLON.NS", "YESBANK.NS",
    "JIOFIN.NS", "ADANIPOWER.NS", "ADANIGREEN.NS",
    "TTML.NS", "HFCL.NS", "NBCC.NS", "NCC.NS",
    "HUDCO.NS", "SJVN.NS", "JSWENERGY.NS",
    "ITC.NS", "SBIN.NS", "TATASTEEL.NS", "WIPRO.NS",
    "PNB.NS", "NHPC.NS", "IRFC.NS", "SAIL.NS", "IOC.NS", "IDEA.NS",
]

# Max watchlist size — keep it manageable for API quotas
MAX_WATCHLIST = 20
MIN_WATCHLIST = 10
WATCHLIST_STATE_FILE = os.path.join(config.PROJECT_DIR, "watchlist_state.json")
CYCLE_COUNT_FILE = os.path.join(config.PROJECT_DIR, "logs", "cycle_count.txt")
MAX_DAILY_DRAWDOWN_PCT = 3.5
SYMBOL_COOLDOWN_MIN = 45
# Rotate a symbol out of the watchlist after this many cycles with no
# actionable BUY/SELL signal from any source (AI or rule-based, raw conf
# >= 0.50). At a 15-min cadence, 8 cycles ≈ 2 hours of dead air.
STALE_CYCLE_LIMIT = 8


def _trend_score(hist: pd.DataFrame) -> float | None:
    """Score short-term trend quality using momentum, volume, and volatility."""
    if hist.empty or len(hist) < 20:
        return None

    close = hist["Close"]
    volume = hist["Volume"]
    price = float(close.iloc[-1])
    if price <= 0:
        return None

    ret_5d = float(((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]) * 100) if len(close) >= 6 and close.iloc[-6] != 0 else 0.0
    ret_20d = float(((close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]) * 100) if close.iloc[-20] != 0 else 0.0
    avg_vol = float(volume.tail(20).mean())
    vol_ratio = float(volume.iloc[-1] / avg_vol) if avg_vol > 0 else 1.0
    daily_ret = close.pct_change().dropna().tail(20)
    vol_pct = float(daily_ret.std() * 100) if not daily_ret.empty else 0.0

    # Reward momentum + participation, penalize unstable moves.
    return (ret_5d * 1.4) + (ret_20d * 0.8) + ((vol_ratio - 1.0) * 6.0) - (vol_pct * 0.9)


def _adjust_confidence(confidence: float, signal: str, ml: dict, ml_mature: bool, regime: str) -> tuple[float, bool]:
    adjusted = confidence
    ml_agrees = True

    if ml and ml_mature:
        ml_dir = ml.get("prediction", "")
        ml_conf = float(ml.get("confidence", 0))
        if signal == "BUY" and ml_dir == "DOWN" and ml_conf > 0.6:
            adjusted *= 0.50  # Strong penalty: ML disagrees with BUY
            ml_agrees = False
        elif signal == "SELL" and ml_dir == "UP" and ml_conf > 0.6:
            adjusted *= 0.50  # Strong penalty: ML disagrees with SELL
            ml_agrees = False
        elif signal == "BUY" and ml_dir == "UP":
            adjusted = min(adjusted * 1.05, 1.0)  # Modest boost for agreement
    elif ml and not ml_mature:
        ml_dir = ml.get("prediction", "?")
        ml_agrees = (signal == "BUY" and ml_dir == "UP") or (signal == "SELL" and ml_dir == "DOWN")

    # Regime nudges — modest only. The hard threshold bump in BEAR already
    # filters weak signals; layering a 0.85x penalty on top was
    # double-punishment and blocked nearly all BUYs.
    if regime == "BULL" and signal == "BUY":
        adjusted = min(adjusted * 1.03, 1.0)

    return max(0.0, min(adjusted, 1.0)), ml_agrees


def _sized_position_pct(ai_size_pct: float, adjusted_conf: float, regime: str, hard_cap: float | None = None) -> float:
    """Conservative sizing relative to the portfolio's hard cap.

    `hard_cap` comes from the active portfolio's `max_position_size_pct` — a
    Rs.10k eval portfolio uses a bigger fraction (25%) so it can actually
    afford 1+ share of the stocks it trades, while main stays at 10%.
    """
    if hard_cap is None:
        hard_cap = config.MAX_POSITION_SIZE_PCT
    conf_frac = 0.5 if adjusted_conf < 0.75 else 0.8 if adjusted_conf < 0.85 else 1.0
    regime_frac = 0.5 if regime == "BEAR" else 1.0
    return max(0.01, min(ai_size_pct, hard_cap * conf_frac, hard_cap * regime_frac, hard_cap))


def scan_trending_stocks(
    held_symbols: set[str] | None = None,
    cycle_num: int = 0,
    stale_counts: dict[str, int] | None = None,
) -> list[str]:
    """Scan NSE stocks for trending movers. Returns list of symbols added.

    Rotation rules:
      * stocks with no actionable signal for STALE_CYCLE_LIMIT cycles are
        eligible for eviction (unless currently held);
      * when the watchlist is at MAX_WATCHLIST and we have a stronger
        candidate than the worst stale member, swap them;
      * single-day price drop > 3% still triggers cold removal as before.
    """
    if held_symbols is None:
        held_symbols = set()
    if stale_counts is None:
        stale_counts = {}

    logger.info("  [SCAN] Scanning NSE for trending stocks...")
    state = read_json(WATCHLIST_STATE_FILE, default={})
    held_cycles = state.get("hold_cycles", {})

    # Backfill missing hold_cycles for symbols that joined via config.WATCHLIST
    # rather than the scanner — without this, the cycles_held>=3 guard never
    # fires for the original 20 (everything looks freshly added).
    current = set(config.WATCHLIST)
    for sym in current:
        held_cycles.setdefault(sym, 0)

    # Score every SCAN_POOL candidate
    results = []
    for sym in SCAN_POOL:
        try:
            hist = get_historical_data(sym, period="30d", interval="1d")
            score = _trend_score(hist)
            if score is None:
                continue
            price = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2])
            chg = float(((price - prev) / prev) * 100) if prev > 0 else 0.0
            vol = float(hist["Volume"].iloc[-1])
            avg_vol = float(hist["Volume"].tail(20).mean())
            vol_ratio = vol / avg_vol if avg_vol > 0 else 1
            max_price = config.INITIAL_CAPITAL * config.MAX_POSITION_SIZE_PCT
            if price <= max_price and score > 2.0 and vol > 0:
                results.append({
                    "sym": sym, "price": price, "chg": chg,
                    "vol_ratio": vol_ratio, "score": score,
                })
        except Exception:
            logger.warning(f"  [SCAN] Skipping {sym}: data fetch failed")
    results.sort(key=lambda x: x["score"], reverse=True)

    # Compute stale list (sorted worst first) — used both for direct removal
    # and for swap-eviction when the watchlist is full.
    stale = sorted(
        (
            (sym, stale_counts.get(sym, 0))
            for sym in current
            if sym not in held_symbols and stale_counts.get(sym, 0) >= STALE_CYCLE_LIMIT
        ),
        key=lambda x: -x[1],
    )

    # Drop stale symbols outright (down to MIN_WATCHLIST floor)
    added: list[str] = []
    for sym, count in stale:
        if len(current) <= MIN_WATCHLIST:
            break
        current.discard(sym)
        held_cycles.pop(sym, None)
        stale_counts.pop(sym, None)
        logger.info(f"  [SCAN] - {sym}: removed (stale {count} cycles)")

    # Add new candidates — first fill empty slots, then swap with remaining
    # stale members when at cap.
    stale_remaining = [s for s, _ in stale if s in current]
    for r in results:
        sym = r["sym"]
        if sym in current or sym in held_symbols:
            continue
        if len(current) < MAX_WATCHLIST:
            current.add(sym)
            added.append(sym)
            held_cycles[sym] = cycle_num
            stale_counts[sym] = 0
            logger.info(f"  [SCAN] + {sym}: Rs.{r['price']:.2f} ({r['chg']:+.1f}%, vol:{r['vol_ratio']:.1f}x, score:{r['score']:.1f})")
        elif stale_remaining:
            evict = stale_remaining.pop(0)
            current.discard(evict)
            held_cycles.pop(evict, None)
            stale_counts.pop(evict, None)
            current.add(sym)
            added.append(sym)
            held_cycles[sym] = cycle_num
            stale_counts[sym] = 0
            logger.info(f"  [SCAN] swap: -{evict} +{sym} (Rs.{r['price']:.2f}, score:{r['score']:.1f})")

    # Legacy cold removal (sharp 1-day drop) — retained as a safety net.
    cold = []
    for sym in list(current):
        if sym in SCAN_POOL and sym not in held_symbols:
            try:
                hist = get_historical_data(sym, period="5d", interval="1d")
                if not hist.empty and len(hist) >= 2:
                    price = hist["Close"].iloc[-1]
                    prev = hist["Close"].iloc[-2]
                    chg = ((price - prev) / prev) * 100
                    hold_since = int(held_cycles.get(sym, 0))
                    cycles_held = max(0, cycle_num - hold_since)
                    if chg < -3 and cycles_held >= 3:
                        cold.append(sym)
            except Exception:
                logger.warning(f"  [SCAN] Could not re-evaluate {sym} for removal")
    for sym in cold:
        if len(current) > MIN_WATCHLIST:
            current.discard(sym)
            held_cycles.pop(sym, None)
            stale_counts.pop(sym, None)
            logger.info(f"  [SCAN] - {sym}: removed (cold, -3%+ daily)")

    config.WATCHLIST = list(current)
    write_json_atomic(WATCHLIST_STATE_FILE, {
        "watchlist": config.WATCHLIST,
        "hold_cycles": held_cycles,
        "stale_counts": stale_counts,
        "updated_at": now_ist().isoformat(),
    })
    logger.info(f"  [SCAN] Watchlist: {len(config.WATCHLIST)} stocks")
    return added


def run_trading_cycle(
    trader: PaperTrader | list[PaperTrader],
    cycle_num: int,
    use_ai: bool = True,
    allow_new_entries: bool = True,
    symbol_cooldown: dict[str, datetime] | None = None,
    stale_counts: dict[str, int] | None = None,
) -> None:
    """Run a single trading cycle against one or more portfolios.

    When `trader` is a list, the same AI/ML signals are applied to every
    portfolio in order — this is how the `eval` portfolio rides alongside
    `main` without paying extra LLM cost per cycle.
    """
    traders: list[PaperTrader] = trader if isinstance(trader, list) else [trader]
    if symbol_cooldown is None:
        symbol_cooldown = {}
    ist = now_ist()
    logger.info(f"\n{'='*60}")
    logger.info(f"  CYCLE #{cycle_num} | {ist.strftime('%Y-%m-%d %H:%M:%S')} IST")
    logger.info(f"{'='*60}")

    # Get current prices
    prices = get_watchlist_prices()
    min_coverage = max(1, int(len(config.WATCHLIST) * 0.5))
    if not prices or len(prices) < min_coverage:
        logger.warning(
            f"  Insufficient price coverage ({len(prices)}/{len(config.WATCHLIST)} stocks). "
            f"Skipping cycle to avoid trading on stale/missing data."
        )
        return

    # Check stop-loss / take-profit first (per portfolio)
    for _tr in traders:
        _tr.refresh_portfolio()
        triggered = _tr.check_stop_loss_take_profit(prices)
        if triggered:
            logger.info(f"  [{_tr.name}] Triggered {len(triggered)} stop-loss/take-profit orders")

    if use_ai:
        # Step 1: AI signals (includes news + trade history learning)
        logger.info("  [1/4] Getting AI signals (technicals + news + learning)...")
        try:
            signals = analyze_watchlist()
        except Exception as e:
            logger.warning(f"  [AI] AI unavailable ({e}). Falling back to rule-based signals.")
            signals = []

        # Fall back to rule-based scoring when AI returns nothing OR returns
        # all HOLD. Some weaker free-tier models default the entire watchlist
        # to HOLD which silently freezes both portfolios for days at a time.
        actionable = [s for s in signals if s.get("signal") in ("BUY", "SELL")]
        if not actionable:
            if signals:
                logger.info(f"  [AI] All {len(signals)} signals were HOLD — running rule-based fallback for actionable trades.")
            else:
                logger.info("  [AI] No AI signals returned. Running multi-indicator fallback...")
            signals = []  # rebuild from rule engine below
            for symbol in config.WATCHLIST:
                df = get_historical_data(symbol, period="30d", interval="1d")
                if df.empty:
                    continue
                sig = get_scored_signal(symbol, df)
                if sig["signal"] in ("BUY", "SELL") and sig["confidence"] >= 0.55:
                    # Lowered from 0.70 → 0.55 so eval (loose threshold) has
                    # candidates to act on. Each portfolio still applies its
                    # own confidence_threshold downstream, so main remains
                    # conservative while eval gets the extra surface area.
                    size_pct = max(0.02, min(sig["confidence"] * 0.08, 0.06))
                    signals.append({
                        "symbol": symbol,
                        "signal": sig["signal"],
                        "confidence": sig["confidence"],
                        "price": sig.get("price", 0),
                        "reason": f"[Fallback] {sig.get('reason', '')}",
                        "position_size_pct": size_pct,
                    })
            signals.sort(key=lambda s: s["confidence"], reverse=True)
            if signals:
                logger.info(f"  [AI] Fallback produced {len(signals)} scored signals (top: {signals[0]['symbol']} @ {signals[0]['confidence']:.0%})")

        # Step 2: ML predictions (only used if model is mature enough)
        logger.info("  [2/4] Getting ML predictions...")
        ml_predictions = {}
        ml_mature = False
        training_log_path = os.path.join(config.PROJECT_DIR if hasattr(config, 'PROJECT_DIR') else '.', "training_log.json")
        if os.path.exists(training_log_path):
            import json as _json
            with open(training_log_path) as _f:
                tlog = _json.load(_f)
            if tlog:
                latest_accuracy = tlog[-1].get("metrics", {}).get("cv_accuracy", 0)
                ml_mature = latest_accuracy >= 75  # only trust ML above 75% accuracy (learning mode until proven)
                logger.info(f"  [ML] Model accuracy: {latest_accuracy}% — {'ACTIVE (influencing trades)' if ml_mature else 'OBSERVING ONLY (too immature)'}")

        for symbol in config.WATCHLIST:
            df = get_historical_data(symbol, period="60d", interval="1d")
            if not df.empty:
                pred = predict(symbol, df)
                if "error" not in pred:
                    ml_predictions[symbol] = pred

        # Market Regime Check — main keeps strict thresholds; eval and any
        # other portfolio resolve their own regime threshold below.
        regime = get_market_regime()
        regime_key = regime if regime in ("BULL", "BEAR") else "NEUTRAL"
        if regime == "BEAR":
            logger.warning(f"  [REGIME] BEAR Market detected ({config.MARKET_INDEX} below 200-day SMA).")
        elif regime == "BULL":
            logger.info(f"  [REGIME] BULL Market detected ({config.MARKET_INDEX} above 50-day & 200-day SMA).")
        else:
            logger.info(f"  [REGIME] NEUTRAL Market detected.")

        # Step 3: Combine AI + ML and execute trades — fan out to every portfolio
        logger.info("  [3/4] Executing trades...")
        from datetime import datetime, timezone
        SIGNAL_MAX_AGE_SEC = 120
        # Pre-compute stale-signal filter + indicator snapshots once per symbol
        fresh_signals: list[dict] = []
        for sig in signals:
            generated_at = sig.get("generated_at")
            if generated_at:
                try:
                    age = (datetime.now(timezone.utc) - datetime.fromisoformat(generated_at)).total_seconds()
                    if age > SIGNAL_MAX_AGE_SEC:
                        logger.warning(f"  [STALE] {sig.get('symbol')}: signal is {age:.0f}s old (>{SIGNAL_MAX_AGE_SEC}s); skipping")
                        continue
                except (TypeError, ValueError):
                    pass
            fresh_signals.append(sig)

        # Update per-symbol staleness — actionable means BUY/SELL with raw
        # confidence >= 0.50 from any source. Symbols that keep coming back
        # as HOLD (or low-confidence) get rotated out by the scanner.
        if stale_counts is not None:
            actionable_syms = {
                s["symbol"] for s in fresh_signals
                if s.get("signal") in ("BUY", "SELL") and s.get("confidence", 0) >= 0.50
            }
            for sym in config.WATCHLIST:
                if sym in actionable_syms:
                    stale_counts[sym] = 0
                else:
                    stale_counts[sym] = stale_counts.get(sym, 0) + 1

        for active_trader in traders:
            # Per-portfolio confidence threshold — eval runs more aggressive.
            portfolio_thresholds = config.PORTFOLIO_CONFIDENCE_THRESHOLD.get(
                active_trader.name,
                config.PORTFOLIO_CONFIDENCE_THRESHOLD["main"],
            )
            confidence_threshold = portfolio_thresholds.get(regime_key, 0.72)
            # Per-portfolio ML override — eval lets the immature CatBoost
            # actually influence trades so we can measure its live performance.
            ml_override = config.PORTFOLIO_ML_OVERRIDE.get(active_trader.name, False)
            trader_ml_active = ml_mature or ml_override
            ml_status = "ACTIVE" if trader_ml_active else "OBSERVING"
            if ml_override and not ml_mature:
                ml_status += " (override: immature model)"
            logger.info(
                f"  └─ portfolio={active_trader.name} "
                f"(capital=Rs.{active_trader.initial_capital:,.0f}, "
                f"threshold={confidence_threshold:.2f}, ML={ml_status})"
            )
            for sig in fresh_signals:
                active_trader.refresh_portfolio()
                symbol = sig["symbol"]
                signal = sig.get("signal", "HOLD")
                confidence = sig.get("confidence", 0)

                # Cross-validate with ML model — per-portfolio maturity flag
                ml = ml_predictions.get(symbol, {})
                confidence_adj, ml_agrees = _adjust_confidence(confidence, signal, ml, trader_ml_active, regime)

                if confidence_adj < confidence_threshold:
                    continue

                # Get indicator snapshot for learning
                df = get_historical_data(symbol, period="30d", interval="1d")
                indicators = get_snapshot(symbol, df) if not df.empty else {}
                ml_tag = f"ML:{ml.get('prediction','?')}" if not ml_mature else ("ML agrees" if ml_agrees else "ML disagrees")

                if signal == "BUY" and symbol not in active_trader.portfolio.positions:
                    if not allow_new_entries:
                        continue
                    price = prices.get(symbol, sig.get("price", 0))
                    if price > 0:
                        cooldown_ts = symbol_cooldown.get((active_trader.name, symbol))
                        if cooldown_ts and (now_ist() - cooldown_ts).total_seconds() < SYMBOL_COOLDOWN_MIN * 60:
                            logger.info(f"  [RISK] [{active_trader.name}] Cooldown active for {symbol}; skipping re-entry")
                            continue

                        # Pre-entry validation: volume + divergence + resistance check.
                        # All column accesses guard against missing indicators —
                        # raw OHLCV dataframes (no rsi/volume_sma) shouldn't crash
                        # the cycle, just skip the check.
                        if not df.empty and len(df) >= 10:
                            _latest = df.iloc[-1]
                            if "volume_sma" in df.columns:
                                _vol_sma = _latest.get("volume_sma")
                                if pd.notna(_vol_sma) and _vol_sma > 0:
                                    _vol_ratio = _latest["Volume"] / _vol_sma
                                    if _vol_ratio < 0.8:
                                        logger.info(f"  [FILTER] {symbol} BUY blocked: low volume ({_vol_ratio:.1f}x avg)")
                                        continue
                            # RSI divergence guard
                            if "rsi" in df.columns and len(df) >= 5:
                                _rsi_now = _latest.get("rsi")
                                _rsi_5d_val = df["rsi"].iloc[-5]
                                _rsi_5d = _rsi_5d_val if pd.notna(_rsi_5d_val) else None
                                _price_5d = float(df["Close"].iloc[-5])
                                if pd.notna(_rsi_now) and _rsi_5d and _price_5d > 0:
                                    if price > _price_5d and _rsi_now < _rsi_5d - 3:
                                        logger.info(f"  [FILTER] {symbol} BUY blocked: bearish RSI divergence (price up, RSI {_rsi_now:.0f} < {_rsi_5d:.0f})")
                                        continue
                            # Near-resistance guard
                            _high_20d = float(df["High"].tail(20).max())
                            if _high_20d > 0 and ((_high_20d - price) / price) < 0.015:
                                logger.info(f"  [FILTER] {symbol} BUY blocked: within 1.5% of 20D resistance ({_high_20d:.2f})")
                                continue

                        portfolio_cap = active_trader.max_position_size_pct
                        ai_size_pct = float(sig.get("position_size_pct", 0.05))
                        ai_size_pct = _sized_position_pct(
                            max(0.01, min(ai_size_pct, portfolio_cap)),
                            confidence_adj,
                            regime,
                            hard_cap=portfolio_cap,
                        )
                        logger.info(f"  [AI][{active_trader.name}] {sig.get('reason', '')} ({ml_tag})")
                        order = active_trader.buy(
                            symbol,
                            price,
                            confidence=confidence_adj,
                            max_position_size_pct=ai_size_pct,
                            ai_signal=sig,
                        )
                        if order:
                            log_trade(symbol, "BUY", price, order.quantity,
                                      ai_signal=sig, indicators=indicators,
                                      market_context={"ml_prediction": ml},
                                      portfolio=active_trader.name)

                elif signal == "SELL" and symbol in active_trader.portfolio.positions:
                    price = prices.get(symbol, sig.get("price", 0))
                    pos = active_trader.portfolio.positions.get(symbol)
                    if price > 0 and pos:
                        price_d = D(price)
                        pnl = pos.pnl(price_d)
                        pnl_pct = pos.pnl_pct(price_d)
                        entry_time = getattr(pos, "entry_time", None)
                        logger.info(f"  [AI][{active_trader.name}] {sig.get('reason', '')} ({ml_tag})")
                        order = active_trader.sell(symbol, price)
                        if order:
                            symbol_cooldown[(active_trader.name, symbol)] = now_ist()
                            log_trade(symbol, "SELL", price, order.quantity,
                                      ai_signal=sig, indicators=indicators,
                                      market_context={"ml_prediction": ml},
                                      portfolio=active_trader.name)
                            record_outcome(symbol, price, pnl, pnl_pct,
                                           entry_time=entry_time,
                                           portfolio=active_trader.name)

        # Step 4: Update lessons learned (per portfolio)
        logger.info("  [4/4] Updating lessons...")
        for _tr in traders:
            generate_lessons(_tr.name)
    else:
        # Rule-based signals (no API calls) — fan out to each portfolio
        for symbol in config.WATCHLIST:
            df = get_historical_data(symbol, period="30d", interval="1d")
            if df.empty:
                continue
            sig = get_latest_signal(symbol, df)
            for active_trader in traders:
                active_trader.refresh_portfolio()
                if sig["signal"] == "BUY" and symbol not in active_trader.portfolio.positions:
                    price = prices.get(symbol, sig["price"])
                    active_trader.buy(symbol, price)
                elif sig["signal"] == "SELL" and symbol in active_trader.portfolio.positions:
                    price = prices.get(symbol, sig["price"])
                    active_trader.sell(symbol, price)

    # Show status per portfolio
    for active_trader in traders:
        summary = active_trader.get_summary(prices)
        logger.info(f"\n  [{active_trader.name}] Cash: Rs.{summary['cash']:.2f} | "
              f"Positions: Rs.{summary['positions_value']:.2f} | "
              f"Total: Rs.{summary['total_value']:.2f} | "
              f"Return: {summary['total_return_pct']:+.2f}%")

        if active_trader.portfolio.positions:
            for sym, pos in active_trader.portfolio.positions.items():
                current = D(prices.get(sym, pos.avg_price))
                pnl = pos.pnl(current)
                pnl_pct = pos.pnl_pct(current)
                pnl_str = f"+Rs.{pnl:.2f}" if pnl >= 0 else f"-Rs.{abs(pnl):.2f}"
                logger.info(f"    [{active_trader.name}] {sym:20s} {pos.quantity}x @ Rs.{pos.avg_price:.2f} -> Rs.{current:.2f}  {pnl_str} ({pnl_pct:+.1f}%)")


def run_autopilot(interval_min: int = 15, use_ai: bool = True, force: bool = False):
    """
    Run trading bot on autopilot.

    Args:
        interval_min: Minutes between each trading cycle
        use_ai: Use AI/Groq (True) or rule-based strategy (False)
        force: Run even outside market hours (for testing)
    """
    mode = "AI (Gemma 4)" if use_ai else "Rule-based (RSI+EMA)"
    logger.info(f"""
{'='*60}
  AI TRADING AGENT - AUTOPILOT MODE
{'='*60}
  Strategy:    {mode}
  Interval:    Every {interval_min} minutes
  Capital:     Rs.{config.INITIAL_CAPITAL}
  Watchlist:   {len(config.WATCHLIST)} stocks
  Market:      9:15 AM - 3:30 PM IST
  Force mode:  {'ON (runs outside market hours)' if force else 'OFF'}
{'='*60}
  Press Ctrl+C to stop
{'='*60}
""")

    # One trader per configured portfolio — same signals, different capital.
    traders = [PaperTrader(name=name) for name in config.PORTFOLIOS.keys()]
    trader = traders[0]  # keep `trader` alias for legacy references below
    logger.info(f"  Portfolios: {', '.join(f'{t.name}=Rs.{t.initial_capital:,.0f}' for t in traders)}")

    # Restore cycle count across restarts so SCAN_EVERY_N_CYCLES math, the
    # 10-cycle performance report cadence, and per-symbol staleness keep
    # accumulating instead of resetting on every service restart.
    cycle = 0
    try:
        with open(CYCLE_COUNT_FILE) as cf:
            cycle = int(cf.read().strip())
        logger.info(f"  [INIT] Restored cycle counter from disk: {cycle}")
    except (OSError, ValueError):
        pass
    last_train_date = None
    SCAN_EVERY_N_CYCLES = 3  # Every 3 cycles = 45 min at 15-min interval

    # Restore persisted watchlist from last session (scan_trending_stocks saves it)
    saved_state = read_json(WATCHLIST_STATE_FILE, default={})
    saved_watchlist = saved_state.get("watchlist")
    if saved_watchlist and isinstance(saved_watchlist, list) and len(saved_watchlist) >= MIN_WATCHLIST:
        config.WATCHLIST = saved_watchlist
        logger.info(f"  [SCAN] Restored watchlist from last session: {len(config.WATCHLIST)} stocks")
    # Restore per-symbol staleness so the rotation accumulates across restarts.
    stale_counts: dict[str, int] = saved_state.get("stale_counts", {})
    if stale_counts:
        logger.info(f"  [SCAN] Restored staleness for {len(stale_counts)} symbols")
    day_start_equity: float | None = None
    day_ref: datetime.date | None = None
    symbol_cooldown: dict[str, datetime] = {}

    while True:
        try:
            if not force and not is_market_open():
                wait = time_to_market_open()
                if wait:
                    hours = wait.total_seconds() / 3600
                    ist = now_ist()
                    logger.info(f"  [{ist.strftime('%H:%M IST')}] Market closed. Opens in {hours:.1f} hours. Waiting...")
                    sleep_secs = min(wait.total_seconds(), 300)
                    time.sleep(sleep_secs)
                    continue

            today = now_ist().date()

            # Auto re-train ML model once per day (first cycle of the day)
            if use_ai and last_train_date != today:
                logger.info(f"\n  [ML] Daily model re-training...")
                try:
                    do_train, reason = should_retrain(min_hours=18)
                    if do_train:
                        metrics = train_model()
                        if "error" not in metrics:
                            promoted = "PROMOTED" if metrics.get("model_promoted") else "NOT promoted"
                            logger.info(
                                f"  [ML] Trained on {metrics['samples']} samples, "
                                f"WF F1: {metrics.get('walk_forward_f1', 0)}%, "
                                f"Holdout F1: {metrics.get('holdout_f1', 0)}% ({promoted})"
                            )
                        else:
                            logger.warning(f"  [ML] Training skipped: {metrics['error']}")
                    else:
                        logger.info(f"  [ML] Retraining skipped: {reason}")
                except Exception as e:
                    logger.error(f"  [ML] Training failed: {e}")
                last_train_date = today

            # Hot-reload config overrides (watchlist, risk params, etc.)
            changed = config.reload_overrides()
            if changed:
                logger.info(f"  [CONFIG] Hot-reloaded: {', '.join(changed)}")

            cycle += 1
            # Persist cycle count so it survives restarts and the API can
            # read it without log parsing.
            try:
                with open(CYCLE_COUNT_FILE, "w") as cf:
                    cf.write(str(cycle))
            except Exception:
                pass

            # Trending stock scan: every 3 cycles (45 min) — uses staleness
            # tracking to evict symbols that have stopped producing signals.
            if cycle % SCAN_EVERY_N_CYCLES == 0 or not config.WATCHLIST:
                try:
                    logger.info(f"  [SCAN] Watchlist update (cycle {cycle})...")
                    scan_trending_stocks(
                        held_symbols=set(trader.portfolio.positions.keys()),
                        cycle_num=cycle,
                        stale_counts=stale_counts,
                    )
                except Exception as e:
                    logger.error(f"  [SCAN] Error: {e}")
            prices = get_watchlist_prices()
            allow_new_entries = True
            if prices:
                if day_ref != today:
                    day_ref = today
                    day_start_equity = trader.get_summary(prices)["total_value"]
                if day_start_equity:
                    current_equity = trader.get_summary(prices)["total_value"]
                    day_return_pct = ((current_equity - day_start_equity) / day_start_equity) * 100
                    if day_return_pct <= -MAX_DAILY_DRAWDOWN_PCT:
                        allow_new_entries = False
                        logger.warning(
                            f"  [RISK] Daily drawdown {day_return_pct:.2f}% <= -{MAX_DAILY_DRAWDOWN_PCT:.2f}%. "
                            "Blocking new BUY entries for today."
                        )
            run_trading_cycle(
                traders,
                cycle,
                use_ai=use_ai,
                allow_new_entries=allow_new_entries,
                symbol_cooldown=symbol_cooldown,
                stale_counts=stale_counts,
            )

            # Show daily performance report every 10 cycles (per portfolio)
            if cycle % 10 == 0:
                for _tr in traders:
                    print_performance_report(_tr.name)

            # Sleep until next cycle
            ist = now_ist()
            logger.info(f"\n  Next cycle in {interval_min} minutes... (Ctrl+C to stop)")
            time.sleep(interval_min * 60)

        except KeyboardInterrupt:
            logger.info("\n\n  Autopilot stopped by user.")
            # Final summary with full report (per portfolio)
            prices = get_watchlist_prices()
            if prices:
                for _tr in traders:
                    summary = _tr.get_summary(prices)
                    logger.info(f"\n  FINAL STATUS [{_tr.name}]:")
                    logger.info(f"  Total Value: Rs.{summary['total_value']:.2f}")
                    logger.info(f"  Return: {summary['total_return_pct']:+.2f}%")
                    logger.info(f"  Trades: {summary.get('total_trades', '?')}")
            for _tr in traders:
                print_performance_report(_tr.name)
            break
        except Exception as e:
            logger.error(f"  Unhandled autopilot error: {e}")
            logger.info("  Sleeping 60s before retry...")
            time.sleep(60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Trading Agent Autopilot")
    parser.add_argument("--interval", type=int, default=15, help="Minutes between cycles (default: 15)")
    parser.add_argument("--no-ai", action="store_true", help="Use rule-based strategy instead of AI")
    parser.add_argument("--force", action="store_true", help="Run outside market hours (for testing)")
    args = parser.parse_args()

    run_autopilot(
        interval_min=args.interval,
        use_ai=not args.no_ai,
        force=args.force,
    )
