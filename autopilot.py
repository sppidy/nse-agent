"""Autopilot mode — runs the AI trading bot continuously during market hours."""

import os
import time
import sys
from datetime import datetime, timedelta

import yfinance as yf

import config
from data_fetcher import get_watchlist_prices, get_historical_data, get_market_regime
from paper_trader import PaperTrader
from ai_strategy import analyze_watchlist, get_portfolio_advice
from strategy import get_latest_signal
from learner import log_trade, record_outcome, get_snapshot, generate_lessons, print_performance_report
from predictor import predict, train_model
from logger import logger


# IST offset from UTC
IST_OFFSET = timedelta(hours=5, minutes=30)

# Market hours (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MIN = 30


def now_ist() -> datetime:
    """Get current time in IST."""
    from datetime import timezone
    utc_now = datetime.now(timezone.utc)
    return utc_now + IST_OFFSET


def is_market_open() -> bool:
    """Check if Indian stock market is currently open."""
    t = now_ist()
    # Weekday check (Mon=0, Fri=4)
    if t.weekday() > 4:
        return False
    market_open = t.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN, second=0)
    market_close = t.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MIN, second=0)
    return market_open <= t <= market_close


def time_to_market_open() -> timedelta | None:
    """Get time remaining until market opens. None if open."""
    if is_market_open():
        return None
    t = now_ist()
    today_open = t.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN, second=0, microsecond=0)
    if t < today_open and t.weekday() <= 4:
        return today_open - t
    # Next weekday
    days_ahead = 1
    while True:
        next_day = t + timedelta(days=days_ahead)
        if next_day.weekday() <= 4:
            next_open = next_day.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN, second=0, microsecond=0)
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


def scan_trending_stocks() -> list[str]:
    """Scan NSE stocks for trending movers. Returns list of symbols to add."""
    logger.info("  [SCAN] Scanning NSE for trending stocks...")
    results = []
    for sym in SCAN_POOL:
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="5d")
            if hist.empty or len(hist) < 2:
                continue
            price = hist["Close"].iloc[-1]
            prev = hist["Close"].iloc[-2]
            chg = ((price - prev) / prev) * 100
            vol = hist["Volume"].iloc[-1]
            avg_vol = hist["Volume"].mean()
            vol_ratio = vol / avg_vol if avg_vol > 0 else 1
            # Filter: affordable + trending
            max_price = config.INITIAL_CAPITAL * config.MAX_POSITION_SIZE_PCT
            if price <= max_price and (chg > 3 or (chg > 1.5 and vol_ratio > 1.3)):
                results.append({
                    "sym": sym, "price": price, "chg": chg,
                    "vol_ratio": vol_ratio, "score": chg + (vol_ratio - 1) * 2,
                })
        except Exception:
            pass

    # Sort by score (change % + volume bonus)
    results.sort(key=lambda x: x["score"], reverse=True)

    # Update watchlist
    current = set(config.WATCHLIST)
    added = []
    for r in results:
        if r["sym"] not in current and len(current) < MAX_WATCHLIST:
            current.add(r["sym"])
            added.append(r["sym"])
            logger.info(f"  [SCAN] + {r['sym']}: Rs.{r['price']:.2f} ({r['chg']:+.1f}%, vol:{r['vol_ratio']:.1f}x)")

    # Remove stocks that have gone cold (change < -2% and not in positions)
    cold = []
    held_symbols = set()  # Will be updated with actual positions in the caller
    for sym in list(current):
        if sym in SCAN_POOL:
            try:
                t = yf.Ticker(sym)
                hist = t.history(period="5d")
                if not hist.empty and len(hist) >= 2:
                    price = hist["Close"].iloc[-1]
                    prev = hist["Close"].iloc[-2]
                    chg = ((price - prev) / prev) * 100
                    if chg < -3 and sym not in held_symbols:
                        cold.append(sym)
            except Exception:
                pass

    for sym in cold:
        if len(current) > 10:  # Keep at least 10 stocks
            current.discard(sym)
            logger.info(f"  [SCAN] - {sym}: removed (cold)")

    config.WATCHLIST = list(current)
    logger.info(f"  [SCAN] Watchlist: {len(config.WATCHLIST)} stocks")
    return added


def run_trading_cycle(trader: PaperTrader, cycle_num: int, use_ai: bool = True) -> None:
    """Run a single trading cycle."""
    ist = now_ist()
    logger.info(f"\n{'='*60}")
    logger.info(f"  CYCLE #{cycle_num} | {ist.strftime('%Y-%m-%d %H:%M:%S')} IST")
    logger.info(f"{'='*60}")

    # Get current prices
    prices = get_watchlist_prices()
    if not prices:
        logger.warning("  Could not fetch prices. Skipping cycle.")
        return

    # Check stop-loss / take-profit first
    triggered = trader.check_stop_loss_take_profit(prices)
    if triggered:
        logger.info(f"  Triggered {len(triggered)} stop-loss/take-profit orders")

    if use_ai:
        # Step 1: AI signals (includes news + trade history learning)
        logger.info("  [1/4] Getting AI signals (technicals + news + learning)...")
        signals = analyze_watchlist()

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
                ml_mature = latest_accuracy >= 55  # only trust ML above 55% accuracy
                logger.info(f"  [ML] Model accuracy: {latest_accuracy}% — {'ACTIVE (influencing trades)' if ml_mature else 'OBSERVING ONLY (too immature)'}")

        for symbol in config.WATCHLIST:
            df = get_historical_data(symbol, period="60d", interval="1d")
            if not df.empty:
                pred = predict(symbol, df)
                if "error" not in pred:
                    ml_predictions[symbol] = pred

        # Market Regime Check
        regime = get_market_regime()
        original_max_position = config.MAX_POSITION_SIZE_PCT
        confidence_threshold = 0.6
        if regime == "BEAR":
            logger.warning(f"  [REGIME] BEAR Market detected ({config.MARKET_INDEX} below 200-day SMA). Halving position sizes and increasing confidence threshold.")
            config.MAX_POSITION_SIZE_PCT = original_max_position / 2
            confidence_threshold = 0.8
        elif regime == "BULL":
            logger.info(f"  [REGIME] BULL Market detected ({config.MARKET_INDEX} above 50-day & 200-day SMA). Normal trading rules apply.")
        else:
            logger.info(f"  [REGIME] NEUTRAL Market detected. Normal trading rules apply.")

        # Step 3: Combine AI + ML and execute trades
        logger.info("  [3/4] Executing trades...")
        for sig in signals:
            symbol = sig["symbol"]
            signal = sig.get("signal", "HOLD")
            confidence = sig.get("confidence", 0)

            # Cross-validate with ML model — only if mature
            ml = ml_predictions.get(symbol, {})
            ml_agrees = True
            if ml and ml_mature:
                ml_dir = ml.get("prediction", "")
                if signal == "BUY" and ml_dir == "DOWN" and ml.get("confidence", 0) > 0.6:
                    confidence *= 0.7  # reduce confidence if ML disagrees
                    ml_agrees = False
                elif signal == "SELL" and ml_dir == "UP" and ml.get("confidence", 0) > 0.6:
                    confidence *= 0.7
                    ml_agrees = False
                elif signal == "BUY" and ml_dir == "UP":
                    confidence = min(confidence * 1.1, 1.0)  # boost if both agree
            elif ml and not ml_mature:
                # Log ML prediction for tracking but don't influence trades
                ml_dir = ml.get("prediction", "?")
                ml_agrees = (signal == "BUY" and ml_dir == "UP") or (signal == "SELL" and ml_dir == "DOWN")

            if confidence < confidence_threshold:
                continue

            # Get indicator snapshot for learning
            df = get_historical_data(symbol, period="30d", interval="1d")
            indicators = get_snapshot(symbol, df) if not df.empty else {}
            ml_tag = f"ML:{ml.get('prediction','?')}" if not ml_mature else ("ML agrees" if ml_agrees else "ML disagrees")

            if signal == "BUY" and symbol not in trader.portfolio.positions:
                price = prices.get(symbol, sig.get("price", 0))
                if price > 0:
                    logger.info(f"  [AI] {sig.get('reason', '')} ({ml_tag})")
                    order = trader.buy(symbol, price, confidence=confidence)
                    if order:
                        log_trade(symbol, "BUY", price, order.quantity,
                                  ai_signal=sig, indicators=indicators,
                                  market_context={"ml_prediction": ml})

            elif signal == "SELL" and symbol in trader.portfolio.positions:
                price = prices.get(symbol, sig.get("price", 0))
                pos = trader.portfolio.positions.get(symbol)
                if price > 0 and pos:
                    pnl = pos.pnl(price)
                    pnl_pct = pos.pnl_pct(price)
                    logger.info(f"  [AI] {sig.get('reason', '')} ({ml_tag})")
                    order = trader.sell(symbol, price)
                    if order:
                        log_trade(symbol, "SELL", price, order.quantity,
                                  ai_signal=sig, indicators=indicators,
                                  market_context={"ml_prediction": ml})
                        record_outcome(symbol, price, pnl, pnl_pct)

        # Restore config
        config.MAX_POSITION_SIZE_PCT = original_max_position

        # Step 4: Update lessons learned
        logger.info("  [4/4] Updating lessons...")
        generate_lessons()
    else:
        # Rule-based signals (no API calls)
        for symbol in config.WATCHLIST:
            df = get_historical_data(symbol, period="30d", interval="1d")
            if df.empty:
                continue
            sig = get_latest_signal(symbol, df)

            if sig["signal"] == "BUY" and symbol not in trader.portfolio.positions:
                price = prices.get(symbol, sig["price"])
                trader.buy(symbol, price)
            elif sig["signal"] == "SELL" and symbol in trader.portfolio.positions:
                price = prices.get(symbol, sig["price"])
                trader.sell(symbol, price)

    # Show status
    summary = trader.get_summary(prices)
    logger.info(f"\n  Cash: Rs.{summary['cash']:.2f} | "
          f"Positions: Rs.{summary['positions_value']:.2f} | "
          f"Total: Rs.{summary['total_value']:.2f} | "
          f"Return: {summary['total_return_pct']:+.2f}%")

    if trader.portfolio.positions:
        for sym, pos in trader.portfolio.positions.items():
            current = prices.get(sym, pos.avg_price)
            pnl = pos.pnl(current)
            pnl_pct = pos.pnl_pct(current)
            pnl_str = f"+Rs.{pnl:.2f}" if pnl >= 0 else f"-Rs.{abs(pnl):.2f}"
            logger.info(f"    {sym:20s} {pos.quantity}x @ Rs.{pos.avg_price:.2f} -> Rs.{current:.2f}  {pnl_str} ({pnl_pct:+.1f}%)")


def run_autopilot(interval_min: int = 15, use_ai: bool = True, force: bool = False):
    """
    Run trading bot on autopilot.

    Args:
        interval_min: Minutes between each trading cycle
        use_ai: Use Gemini AI (True) or rule-based strategy (False)
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

    trader = PaperTrader()
    cycle = 0
    last_train_date = None
    SCAN_EVERY_N_CYCLES = 3  # Every 3 cycles = 45 min at 15-min interval

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
                    metrics = train_model()
                    if "error" not in metrics:
                        logger.info(f"  [ML] Trained on {metrics['samples']} samples, accuracy: {metrics['cv_accuracy']}%")
                    else:
                        logger.warning(f"  [ML] Training skipped: {metrics['error']}")
                except Exception as e:
                    logger.error(f"  [ML] Training failed: {e}")
                last_train_date = today

            cycle += 1

            # Trending stock scan: at market start (cycle 1) then every 3 cycles (45 min)
            if cycle == 1 or cycle % SCAN_EVERY_N_CYCLES == 0:
                try:
                    logger.info(f"  [SCAN] Watchlist update (cycle {cycle})...")
                    scan_trending_stocks()
                except Exception as e:
                    logger.error(f"  [SCAN] Error: {e}")
            run_trading_cycle(trader, cycle, use_ai=use_ai)

            # Show daily performance report every 10 cycles
            if cycle % 10 == 0:
                print_performance_report()

            # Sleep until next cycle
            ist = now_ist()
            logger.info(f"\n  Next cycle in {interval_min} minutes... (Ctrl+C to stop)")
            time.sleep(interval_min * 60)

        except KeyboardInterrupt:
            logger.info("\n\n  Autopilot stopped by user.")
            # Final summary with full report
            prices = get_watchlist_prices()
            if prices:
                summary = trader.get_summary(prices)
                logger.info(f"\n  FINAL STATUS:")
                logger.info(f"  Total Value: Rs.{summary['total_value']:.2f}")
                logger.info(f"  Return: {summary['total_return_pct']:+.2f}%")
                logger.info(f"  Trades: {summary['total_trades']}")
            print_performance_report()
            break


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
