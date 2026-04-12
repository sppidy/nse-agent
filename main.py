"""AI Trading Agent — Main entry point."""

import sys
import time
from tabulate import tabulate

import config
from data_fetcher import get_historical_data, get_live_price, get_watchlist_prices
from paper_trader import PaperTrader, Portfolio
from strategy import get_latest_signal, generate_signals
from backtester import backtest, backtest_portfolio, print_backtest_report
from ai_strategy import analyze_single_stock, analyze_watchlist, get_portfolio_advice


def cmd_ai_scan():
    """Use Gemini AI to analyze watchlist stocks."""
    print(f"\n[AI] Analyzing {len(config.WATCHLIST)} stocks with Gemini...\n")
    signals = analyze_watchlist()

    for sig in signals:
        indicator = {"BUY": ">>", "SELL": "<<", "HOLD": "  "}.get(sig.get("signal", "HOLD"), "  ")
        confidence = sig.get("confidence", 0)
        size_pct = sig.get("position_size_pct", confidence) * 100
        conf_bar = "#" * int(confidence * 10) + "." * (10 - int(confidence * 10))
        print(f"  {indicator} {sig['symbol']:20s}  Rs.{sig.get('price', 0):>10.2f}  {sig.get('signal', 'HOLD'):4s}  [{conf_bar}] {confidence:.0%}  Size:{size_pct:.0f}%")
        print(f"     {sig.get('reason', '')}")
        if sig.get("stop_loss"):
            print(f"     SL: Rs.{sig['stop_loss']:.2f}  Target: Rs.{sig.get('target', 0):.2f}")
        print()

    buys = [s for s in signals if s.get("signal") == "BUY"]
    sells = [s for s in signals if s.get("signal") == "SELL"]
    print(f"  {len(buys)} BUY, {len(sells)} SELL, {len(signals) - len(buys) - len(sells)} HOLD")
    return signals


def cmd_ai_trade():
    """Run one AI-powered paper trading cycle."""
    trader = PaperTrader()
    print("\n[AI] Running Gemini-powered trading cycle...\n")

    # Get current prices
    prices = get_watchlist_prices()
    if not prices:
        print("Could not fetch prices. Market may be closed.")
        return

    # Check stop-loss / take-profit first
    trader.check_stop_loss_take_profit(prices)

    # Get AI signals
    signals = analyze_watchlist()

    for sig in signals:
        symbol = sig["symbol"]
        signal = sig.get("signal", "HOLD")
        confidence = sig.get("confidence", 0)

        # Only act on high-confidence signals
        if confidence < 0.6:
            continue

        if signal == "BUY" and symbol not in trader.portfolio.positions:
            price = prices.get(symbol, sig.get("price", 0))
            if price > 0:
                print(f"  [AI] {sig.get('reason', '')}")
                trader.buy(symbol, price)

        elif signal == "SELL" and symbol in trader.portfolio.positions:
            price = prices.get(symbol, sig.get("price", 0))
            if price > 0:
                print(f"  [AI] {sig.get('reason', '')}")
                trader.sell(symbol, price)

    # Get portfolio advice
    summary = trader.get_summary(prices)
    print("\n[AI] Portfolio Advice:")
    print("-" * 50)
    advice = get_portfolio_advice(summary, signals)
    print(advice)
    print("-" * 50)

    cmd_status(trader, prices)


def cmd_scan():
    """Scan watchlist for trading signals."""
    print(f"\nScanning {len(config.WATCHLIST)} stocks...\n")
    signals = []
    for symbol in config.WATCHLIST:
        df = get_historical_data(symbol, period="60d", interval="1d")
        if df.empty:
            continue
        sig = get_latest_signal(symbol, df)
        signals.append(sig)
        indicator = {"BUY": ">>", "SELL": "<<", "HOLD": "  "}[sig["signal"]]
        print(f"  {indicator} {symbol:20s}  Rs.{sig['price']:>10.2f}  {sig['signal']:4s}  {sig['reason']}")

    buys = [s for s in signals if s["signal"] == "BUY"]
    sells = [s for s in signals if s["signal"] == "SELL"]
    print(f"\n  {len(buys)} BUY signals, {len(sells)} SELL signals, {len(signals) - len(buys) - len(sells)} HOLD")
    return signals


def cmd_backtest(symbols=None, period="60d"):
    """Run backtest on watchlist or specific symbols."""
    if symbols is None:
        symbols = config.WATCHLIST
    print(f"\nBacktesting {len(symbols)} stocks over {period}...\n")
    results = []
    data_by_symbol = {}
    for symbol in symbols:
        df = get_historical_data(symbol, period=period, interval="1d")
        if df.empty:
            results.append({"symbol": symbol, "error": "No data"})
            continue
        data_by_symbol[symbol] = df
        result = backtest(symbol, df)
        results.append(result)
    print_backtest_report(results)

    portfolio_result = backtest_portfolio(data_by_symbol, initial_capital=config.INITIAL_CAPITAL)
    if "error" not in portfolio_result:
        print("\nPORTFOLIO-LEVEL BACKTEST (shared capital)")
        print("-" * 50)
        print(f"Initial Capital: Rs.{portfolio_result['initial_capital']:.2f}")
        print(f"Final Equity:    Rs.{portfolio_result['final_equity']:.2f}")
        print(f"Return:          {portfolio_result['total_return_pct']:+.2f}%")
        print(f"Trades:          {portfolio_result['total_trades']}  | Win Rate: {portfolio_result['win_rate']:.1f}%")
        print(f"Realized P&L:    Rs.{portfolio_result['total_pnl']:.2f}")
        print(f"Open Positions:  {portfolio_result['open_positions']}")
    return results


def cmd_trade():
    """Run one cycle of the paper trading bot."""
    trader = PaperTrader()
    print("\nRunning trading cycle...")

    # Get current prices
    prices = get_watchlist_prices()
    if not prices:
        print("Could not fetch prices. Market may be closed.")
        return

    # Check stop-loss / take-profit on existing positions
    trader.check_stop_loss_take_profit(prices)

    # Scan for signals
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

    # Print summary
    cmd_status(trader, prices)


def cmd_status(trader=None, prices=None):
    """Show portfolio status."""
    if trader is None:
        trader = PaperTrader()
    if prices is None:
        prices = get_watchlist_prices()

    summary = trader.get_summary(prices)

    # Find last traded time from orders
    last_trade_time = "No trades yet"
    if trader.portfolio.orders:
        last_ts = trader.portfolio.orders[-1].get("timestamp", "")
        if last_ts:
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(last_ts)
                last_trade_time = dt.strftime("%d-%b-%Y %I:%M:%S %p")
            except ValueError:
                last_trade_time = last_ts

    print(f"\n{'='*50}")
    print(f"  PORTFOLIO STATUS")
    print(f"{'='*50}")
    print(f"  Cash:            Rs.{summary['cash']:>10.2f}")
    print(f"  Positions Value: Rs.{summary['positions_value']:>10.2f}")
    print(f"  Total Value:     Rs.{summary['total_value']:>10.2f}")
    print(f"  Initial Capital: Rs.{summary['initial_capital']:>10.2f}")
    print(f"  Total Return:     {summary['total_return_pct']:>9.2f}%")
    print(f"  Realized P&L:    Rs.{summary['realized_pnl']:>10.2f}")
    print(f"  Open Positions:   {summary['open_positions']:>9d}")
    print(f"  Total Trades:     {summary['total_trades']:>9d}")
    print(f"  Last Traded:      {last_trade_time}")
    print(f"{'='*50}")

    if trader.portfolio.positions:
        print(f"\n  Open Positions:")
        for sym, pos in trader.portfolio.positions.items():
            current = prices.get(sym, pos.avg_price)
            pnl = pos.pnl(current)
            pnl_pct = pos.pnl_pct(current)
            pnl_str = f"+Rs.{pnl:.2f}" if pnl >= 0 else f"-Rs.{abs(pnl):.2f}"
            print(f"    {sym:20s} {pos.quantity}x @ Rs.{pos.avg_price:.2f} -> Rs.{current:.2f}  {pnl_str} ({pnl_pct:+.1f}%)")

    if trader.portfolio.trade_log:
        print(f"\n  Recent Trades:")
        for t in trader.portfolio.trade_log[-5:]:
            pnl_str = f"+Rs.{t['pnl']:.2f}" if t["pnl"] >= 0 else f"-Rs.{abs(t['pnl']):.2f}"
            trade_time = ""
            if t.get("timestamp"):
                from datetime import datetime
                try:
                    dt = datetime.fromisoformat(t["timestamp"])
                    trade_time = dt.strftime(" @ %I:%M %p %d-%b")
                except ValueError:
                    pass
            print(f"    {t['symbol']:20s} {t['quantity']}x  {pnl_str} ({t['pnl_pct']:+.1f}%){trade_time}")


def cmd_news():
    """Fetch and analyze news sentiment."""
    from news_sentiment import print_sentiment_report
    print_sentiment_report()


def cmd_train():
    """Train the local ML prediction model."""
    from predictor import train_model
    print("\nTraining ML model on historical data...\n")
    metrics = train_model()
    if "error" in metrics:
        print(f"  Error: {metrics['error']}")
        return
    print(f"\n  Training complete!")
    print(f"  Samples: {metrics['samples']} | Symbols: {metrics['symbols']}")
    print(f"  Cross-validation accuracy: {metrics['cv_accuracy']}%")
    print(f"  Walk-forward: Acc {metrics.get('walk_forward_accuracy', 0)}% | "
          f"Precision {metrics.get('walk_forward_precision', 0)}% | "
          f"Recall {metrics.get('walk_forward_recall', 0)}% | "
          f"F1 {metrics.get('walk_forward_f1', 0)}%")
    print(f"  Holdout: Acc {metrics.get('holdout_accuracy', 0)}% | "
          f"Precision {metrics.get('holdout_precision', 0)}% | "
          f"Recall {metrics.get('holdout_recall', 0)}% | "
          f"F1 {metrics.get('holdout_f1', 0)}%")
    print(f"  Baseline accuracy: {metrics.get('baseline_accuracy', 0)}% "
          f"(UP class ratio: {metrics.get('up_class_ratio', 0)}%)")
    print(f"  Model promoted: {metrics.get('model_promoted', True)} "
          f"({metrics.get('promotion_reason', 'n/a')})")
    print(f"  CV scores: {metrics['cv_scores']}")
    print(f"  Top features: {', '.join(metrics['top_features'].keys())}")
    print(f"  Model saved to: {metrics['model_path']}")


def cmd_predict():
    """Get ML predictions for watchlist stocks."""
    from predictor import predict_watchlist, print_predictions
    predictions = predict_watchlist()
    print_predictions(predictions)


def cmd_report():
    """Show learning report and performance analytics."""
    from learner import print_performance_report, generate_lessons
    generate_lessons()
    print_performance_report()


def cmd_chat():
    """Start interactive chat with the trading agent."""
    from chat import chat
    chat()


def cmd_reset():
    """Reset portfolio to initial state."""
    Portfolio().save()
    print(f"Portfolio reset to Rs.{config.INITIAL_CAPITAL:.2f}")


def cmd_help():
    print("""
AI Trading Agent — Paper Trading System
========================================

Commands:
  scan      Scan watchlist for BUY/SELL signals (RSI+EMA)
  ai-scan   Scan watchlist using Gemini AI analysis
  backtest  Run backtest on historical data
  trade     Execute one trading cycle (RSI+EMA rules)
  ai-trade  Execute one AI-powered trading cycle (Gemini)
  autopilot Run AI bot all day (auto-trades during market hours)
  chat      Interactive chat with your AI trading agent
  news      Fetch & analyze news sentiment for all stocks
  train     Train local ML model on historical data
  predict   Get ML predictions for tomorrow
  status    Show portfolio and P&L
  report    Show learning report (win rate, patterns, lessons)
  reset     Reset portfolio to configured initial capital
  help      Show this message

Examples:
  python main.py ai-scan                   # AI-powered scan
  python main.py ai-trade                  # Single AI trade cycle
  python main.py autopilot                 # Run all day (every 15 min)
  python main.py autopilot --interval 5    # Run every 5 min
  python main.py autopilot --force         # Run even outside market hours
  python main.py autopilot --no-ai         # Rule-based, no API needed
""")


def main():
    if len(sys.argv) < 2:
        cmd_help()
        return

    command = sys.argv[1].lower()
    commands = {
        "scan": cmd_scan,
        "ai-scan": cmd_ai_scan,
        "backtest": cmd_backtest,
        "trade": cmd_trade,
        "ai-trade": cmd_ai_trade,
        "news": cmd_news,
        "train": cmd_train,
        "predict": cmd_predict,
        "chat": cmd_chat,
        "status": cmd_status,
        "report": cmd_report,
        "reset": cmd_reset,
        "help": cmd_help,
    }

    if command == "autopilot":
        from autopilot import run_autopilot
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("cmd")
        parser.add_argument("--interval", type=int, default=15)
        parser.add_argument("--no-ai", action="store_true")
        parser.add_argument("--force", action="store_true")
        args = parser.parse_args()
        run_autopilot(args.interval, use_ai=not args.no_ai, force=args.force)
    elif command in commands:
        commands[command]()
    else:
        print(f"Unknown command: {command}")
        cmd_help()


if __name__ == "__main__":
    main()
