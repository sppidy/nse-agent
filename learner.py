"""
Learning engine — tracks trade outcomes, learns patterns, and improves over time.

The idea: every trade is logged with full context (indicators, AI reasoning,
market conditions). Before making new trades, we feed this history to Gemini
so it can learn what worked and what didn't.
"""

import os
from datetime import datetime

import pandas as pd

import config
from persistence import read_json, write_json_atomic
from strategy import add_indicators

JOURNAL_FILE = os.path.join(config.PROJECT_DIR, "trade_journal.json")
LESSONS_FILE = os.path.join(config.PROJECT_DIR, "lessons_learned.json")


def _load_json(filepath: str) -> list:
    return read_json(filepath, default=[])


def _save_json(filepath: str, data: list):
    write_json_atomic(filepath, data)


# ── Trade Journal ──────────────────────────────────────────────

def log_trade(
    symbol: str,
    action: str,  # BUY or SELL
    price: float,
    quantity: int,
    ai_signal: dict | None = None,
    indicators: dict | None = None,
    market_context: dict | None = None,
):
    """Log a trade with full context for later analysis."""
    journal = _load_json(JOURNAL_FILE)
    entry = {
        "id": len(journal) + 1,
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "action": action,
        "price": price,
        "quantity": quantity,
        "ai_signal": ai_signal,
        "indicators": indicators,
        "market_context": market_context,
        "outcome": None,  # filled when position is closed
    }
    journal.append(entry)
    _save_json(JOURNAL_FILE, journal)
    return entry


def record_outcome(symbol: str, exit_price: float, pnl: float, pnl_pct: float):
    """Record the outcome of a closed trade."""
    journal = _load_json(JOURNAL_FILE)

    # Find the most recent BUY for this symbol that has no outcome
    for entry in reversed(journal):
        if entry["symbol"] == symbol and entry["action"] == "BUY" and entry["outcome"] is None:
            entry["outcome"] = {
                "exit_price": exit_price,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "exit_time": datetime.now().isoformat(),
                "result": "WIN" if pnl > 0 else "LOSS",
            }
            break

    _save_json(JOURNAL_FILE, journal)


def get_snapshot(symbol: str, df: pd.DataFrame) -> dict:
    """Capture current indicator snapshot for a symbol."""
    df = add_indicators(df)
    if df.empty:
        return {}
    latest = df.iloc[-1]
    return {
        "price": round(float(latest["Close"]), 2),
        "rsi": round(float(latest["rsi"]), 1) if pd.notna(latest.get("rsi")) else None,
        "ema_short": round(float(latest["ema_short"]), 2) if pd.notna(latest.get("ema_short")) else None,
        "ema_long": round(float(latest["ema_long"]), 2) if pd.notna(latest.get("ema_long")) else None,
        "ema_signal": "bullish" if latest.get("ema_short", 0) > latest.get("ema_long", 0) else "bearish",
        "volume_vs_avg": round(float(latest["Volume"] / latest["volume_sma"]), 2)
            if pd.notna(latest.get("volume_sma")) and latest.get("volume_sma", 0) > 0 else None,
        "day_change_pct": round(float((latest["Close"] - latest["Open"]) / latest["Open"] * 100), 2),
    }


# ── Performance Analytics ──────────────────────────────────────

def get_performance_stats() -> dict:
    """Calculate performance stats from trade journal."""
    journal = _load_json(JOURNAL_FILE)
    completed = [e for e in journal if e.get("outcome")]

    if not completed:
        return {"total_trades": 0, "message": "No completed trades yet"}

    wins = [e for e in completed if e["outcome"]["result"] == "WIN"]
    losses = [e for e in completed if e["outcome"]["result"] == "LOSS"]

    avg_win = sum(e["outcome"]["pnl_pct"] for e in wins) / len(wins) if wins else 0
    avg_loss = sum(e["outcome"]["pnl_pct"] for e in losses) / len(losses) if losses else 0

    # Per-symbol stats
    symbol_stats = {}
    for e in completed:
        sym = e["symbol"]
        if sym not in symbol_stats:
            symbol_stats[sym] = {"wins": 0, "losses": 0, "total_pnl": 0}
        if e["outcome"]["result"] == "WIN":
            symbol_stats[sym]["wins"] += 1
        else:
            symbol_stats[sym]["losses"] += 1
        symbol_stats[sym]["total_pnl"] += e["outcome"]["pnl"]

    # Confidence accuracy — did high confidence signals perform better?
    high_conf = [e for e in completed if e.get("ai_signal", {}).get("confidence", 0) >= 0.7]
    low_conf = [e for e in completed if e.get("ai_signal", {}).get("confidence", 0) < 0.7]
    high_conf_winrate = len([e for e in high_conf if e["outcome"]["result"] == "WIN"]) / len(high_conf) * 100 if high_conf else 0
    low_conf_winrate = len([e for e in low_conf if e["outcome"]["result"] == "WIN"]) / len(low_conf) * 100 if low_conf else 0

    return {
        "total_trades": len(completed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(completed) * 100, 1),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "risk_reward": round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0,
        "total_pnl": round(sum(e["outcome"]["pnl"] for e in completed), 2),
        "high_confidence_winrate": round(high_conf_winrate, 1),
        "low_confidence_winrate": round(low_conf_winrate, 1),
        "symbol_stats": symbol_stats,
        "best_stock": max(symbol_stats, key=lambda s: symbol_stats[s]["total_pnl"]) if symbol_stats else None,
        "worst_stock": min(symbol_stats, key=lambda s: symbol_stats[s]["total_pnl"]) if symbol_stats else None,
    }


# ── AI Learning ────────────────────────────────────────────────

def generate_lessons() -> list[dict]:
    """Analyze trade journal and extract lessons learned."""
    journal = _load_json(JOURNAL_FILE)
    completed = [e for e in journal if e.get("outcome")]

    if len(completed) < 3:
        return [{"lesson": "Need at least 3 completed trades to generate lessons"}]

    lessons = []

    # Lesson 1: Which stocks are profitable vs losing
    stats = get_performance_stats()
    for sym, data in stats.get("symbol_stats", {}).items():
        if data["total_pnl"] < 0 and data["losses"] >= 2:
            lessons.append({
                "type": "AVOID_STOCK",
                "symbol": sym,
                "reason": f"Lost Rs.{abs(data['total_pnl']):.2f} over {data['losses']} losing trades",
                "action": f"Reduce position size or avoid {sym} until trend reverses",
            })
        elif data["total_pnl"] > 0 and data["wins"] >= 2:
            lessons.append({
                "type": "FAVOR_STOCK",
                "symbol": sym,
                "reason": f"Gained Rs.{data['total_pnl']:.2f} over {data['wins']} winning trades",
                "action": f"Continue trading {sym}, consider slightly larger positions",
            })

    # Lesson 2: Confidence threshold effectiveness
    if stats.get("high_confidence_winrate", 0) > stats.get("low_confidence_winrate", 0) + 10:
        lessons.append({
            "type": "CONFIDENCE_THRESHOLD",
            "reason": f"High confidence trades win {stats['high_confidence_winrate']}% vs low confidence {stats['low_confidence_winrate']}%",
            "action": "Only trade on signals with confidence >= 0.7",
        })

    # Lesson 3: RSI patterns in winners vs losers
    winning_rsis = [e["indicators"]["rsi"] for e in completed
                    if e["outcome"]["result"] == "WIN" and e.get("indicators", {}).get("rsi")]
    losing_rsis = [e["indicators"]["rsi"] for e in completed
                   if e["outcome"]["result"] == "LOSS" and e.get("indicators", {}).get("rsi")]

    if winning_rsis and losing_rsis:
        avg_win_rsi = sum(winning_rsis) / len(winning_rsis)
        avg_loss_rsi = sum(losing_rsis) / len(losing_rsis)
        if abs(avg_win_rsi - avg_loss_rsi) > 5:
            lessons.append({
                "type": "RSI_PATTERN",
                "reason": f"Winning trades avg RSI: {avg_win_rsi:.0f}, Losing trades avg RSI: {avg_loss_rsi:.0f}",
                "action": f"Prefer entries when RSI is near {avg_win_rsi:.0f}",
            })

    # Lesson 4: Stop-loss analysis
    stopped = [e for e in completed if e["outcome"]["pnl_pct"] <= -config.STOP_LOSS_PCT * 100]
    if len(stopped) >= 2:
        lessons.append({
            "type": "STOP_LOSS",
            "reason": f"{len(stopped)} trades hit stop-loss",
            "action": "Consider tighter stop-loss or waiting for stronger confirmation before entry",
        })

    _save_json(LESSONS_FILE, lessons)
    return lessons


def get_learning_context() -> str:
    """
    Build a context string from past trades and lessons to feed to Gemini.
    This is the key function — it gives the AI "memory" of what worked.
    """
    stats = get_performance_stats()
    lessons = _load_json(LESSONS_FILE) if os.path.exists(LESSONS_FILE) else generate_lessons()
    journal = _load_json(JOURNAL_FILE)
    completed = [e for e in journal if e.get("outcome")]

    if not completed:
        return "No trade history yet. This is the first trading session."

    context = "LEARNING FROM PAST TRADES:\n\n"

    # Overall stats
    context += f"Performance: {stats['total_trades']} trades, "
    context += f"{stats['win_rate']}% win rate, "
    context += f"Avg win: {stats['avg_win_pct']}%, Avg loss: {stats['avg_loss_pct']}%\n"
    context += f"Total P&L: Rs.{stats['total_pnl']}\n\n"

    # Per-symbol performance
    context += "Stock performance:\n"
    for sym, data in stats.get("symbol_stats", {}).items():
        context += f"  {sym}: {data['wins']}W/{data['losses']}L, P&L: Rs.{data['total_pnl']:.2f}\n"

    # Recent trades (last 10)
    context += "\nRecent trades:\n"
    for e in completed[-10:]:
        sig = e.get("ai_signal", {})
        ind = e.get("indicators", {})
        outcome = e["outcome"]
        context += (
            f"  {e['symbol']} | {e['action']} @ Rs.{e['price']:.2f} | "
            f"RSI:{ind.get('rsi','?')} EMA:{ind.get('ema_signal','?')} | "
            f"AI conf:{sig.get('confidence','?')} | "
            f"Result: {outcome['result']} ({outcome['pnl_pct']:+.1f}%)\n"
        )

    # Lessons
    if lessons and not (len(lessons) == 1 and "Need at least" in lessons[0].get("lesson", "")):
        context += "\nLESSONS LEARNED:\n"
        for lesson in lessons:
            context += f"  - [{lesson.get('type', 'GENERAL')}] {lesson.get('reason', '')} -> {lesson.get('action', '')}\n"

    # Confidence accuracy
    if stats.get("high_confidence_winrate", 0) > 0 or stats.get("low_confidence_winrate", 0) > 0:
        context += f"\nConfidence accuracy: High(>=0.7)={stats['high_confidence_winrate']}% win, "
        context += f"Low(<0.7)={stats['low_confidence_winrate']}% win\n"

    context += "\nUSE THIS HISTORY to improve your predictions. Avoid repeating losing patterns. "
    context += "Favor setups that have historically worked. Adjust confidence scores based on track record.\n"

    return context


def print_performance_report():
    """Print a formatted performance report."""
    stats = get_performance_stats()

    if stats.get("total_trades", 0) == 0:
        print("\n  No completed trades yet. Keep running the bot to build history.")
        return

    print(f"\n{'='*60}")
    print(f"  PERFORMANCE REPORT (Learning Engine)")
    print(f"{'='*60}")
    print(f"  Total Trades:    {stats['total_trades']}")
    print(f"  Win Rate:        {stats['win_rate']}%")
    print(f"  Avg Win:         {stats['avg_win_pct']:+.2f}%")
    print(f"  Avg Loss:        {stats['avg_loss_pct']:+.2f}%")
    print(f"  Risk/Reward:     {stats['risk_reward']:.2f}")
    print(f"  Total P&L:       Rs.{stats['total_pnl']:.2f}")
    print(f"  Best Stock:      {stats.get('best_stock', '-')}")
    print(f"  Worst Stock:     {stats.get('worst_stock', '-')}")
    print(f"  High Conf Win%:  {stats['high_confidence_winrate']}%")
    print(f"  Low Conf Win%:   {stats['low_confidence_winrate']}%")

    print(f"\n  Per-Stock Breakdown:")
    for sym, data in stats.get("symbol_stats", {}).items():
        print(f"    {sym:20s} {data['wins']}W/{data['losses']}L  Rs.{data['total_pnl']:+.2f}")

    lessons = generate_lessons()
    if lessons and not (len(lessons) == 1 and "Need at least" in lessons[0].get("lesson", "")):
        print(f"\n  Lessons Learned:")
        for l in lessons:
            print(f"    [{l.get('type', '?')}] {l.get('action', '')}")

    print(f"{'='*60}")
