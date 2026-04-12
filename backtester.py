"""Backtesting engine — test strategies on historical data."""

import pandas as pd
from tabulate import tabulate

import config
from strategy import generate_signals


def backtest(symbol: str, df: pd.DataFrame, initial_capital: float = None) -> dict:
    """
    Backtest a strategy on historical data for a single symbol.
    Returns performance metrics.
    """
    if initial_capital is None:
        initial_capital = config.INITIAL_CAPITAL

    df = generate_signals(df)
    if df.empty:
        return {"symbol": symbol, "error": "No data"}

    cash = initial_capital
    position = 0
    avg_price = 0.0
    trades = []
    equity_curve = []

    for i, row in df.iterrows():
        price = row["Close"]
        signal = row["signal"]

        # Track equity
        equity = cash + position * price
        equity_curve.append({"date": i, "equity": equity})

        if signal == "BUY" and position == 0:
            # Buy with position sizing
            max_spend = cash * config.MAX_POSITION_SIZE_PCT
            qty = int(max_spend / (price * (1 + config.SLIPPAGE_PCT)))
            if qty > 0:
                fill = price * (1 + config.SLIPPAGE_PCT)
                cost = qty * fill + config.BROKERAGE_PER_ORDER
                cash -= cost
                position = qty
                avg_price = fill
                trades.append({
                    "type": "BUY",
                    "date": str(i),
                    "price": round(fill, 2),
                    "qty": qty,
                })

        elif signal == "SELL" and position > 0:
            fill = price * (1 - config.SLIPPAGE_PCT)
            proceeds = position * fill - config.BROKERAGE_PER_ORDER
            pnl = (fill - avg_price) * position
            cash += proceeds
            trades.append({
                "type": "SELL",
                "date": str(i),
                "price": round(fill, 2),
                "qty": position,
                "pnl": round(pnl, 2),
            })
            position = 0

        # Stop loss / take profit
        elif position > 0:
            pnl_pct = (price - avg_price) / avg_price
            if pnl_pct <= -config.STOP_LOSS_PCT:
                fill = price * (1 - config.SLIPPAGE_PCT)
                proceeds = position * fill - config.BROKERAGE_PER_ORDER
                pnl = (fill - avg_price) * position
                cash += proceeds
                trades.append({
                    "type": "STOP_LOSS",
                    "date": str(i),
                    "price": round(fill, 2),
                    "qty": position,
                    "pnl": round(pnl, 2),
                })
                position = 0
            elif pnl_pct >= config.TAKE_PROFIT_PCT:
                fill = price * (1 - config.SLIPPAGE_PCT)
                proceeds = position * fill - config.BROKERAGE_PER_ORDER
                pnl = (fill - avg_price) * position
                cash += proceeds
                trades.append({
                    "type": "TAKE_PROFIT",
                    "date": str(i),
                    "price": round(fill, 2),
                    "qty": position,
                    "pnl": round(pnl, 2),
                })
                position = 0

    # Close any remaining position at last price
    final_price = df.iloc[-1]["Close"]
    if position > 0:
        fill = final_price * (1 - config.SLIPPAGE_PCT)
        proceeds = position * fill
        pnl = (fill - avg_price) * position
        cash += proceeds
        trades.append({
            "type": "CLOSE",
            "date": str(df.index[-1]),
            "price": round(fill, 2),
            "qty": position,
            "pnl": round(pnl, 2),
        })
        position = 0

    final_equity = cash
    total_return = ((final_equity - initial_capital) / initial_capital) * 100
    winning = [t for t in trades if t.get("pnl", 0) > 0]
    losing = [t for t in trades if t.get("pnl", 0) < 0]
    sell_trades = [t for t in trades if t["type"] != "BUY"]

    return {
        "symbol": symbol,
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return, 2),
        "total_trades": len(sell_trades),
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "win_rate": round(len(winning) / len(sell_trades) * 100, 1) if sell_trades else 0,
        "total_pnl": round(sum(t.get("pnl", 0) for t in trades), 2),
        "trades": trades,
        "equity_curve": equity_curve,
    }


def backtest_portfolio(data_by_symbol: dict[str, pd.DataFrame], initial_capital: float | None = None) -> dict:
    """
    Backtest multiple symbols with shared capital.
    Uses independent per-symbol signals but one portfolio cash balance.
    """
    if initial_capital is None:
        initial_capital = config.INITIAL_CAPITAL

    prepped = {}
    all_dates = set()
    for symbol, df in data_by_symbol.items():
        sig_df = generate_signals(df)
        if sig_df.empty:
            continue
        prepped[symbol] = sig_df
        all_dates.update(sig_df.index.tolist())

    if not prepped:
        return {"error": "No data"}

    cash = float(initial_capital)
    positions: dict[str, dict] = {}
    trades = []

    for d in sorted(all_dates):
        for symbol, df in prepped.items():
            if d not in df.index:
                continue
            row = df.loc[d]
            price = float(row["Close"])
            signal = row["signal"]
            pos = positions.get(symbol)

            if signal == "BUY" and pos is None:
                max_spend = cash * config.MAX_POSITION_SIZE_PCT
                qty = int(max_spend / (price * (1 + config.SLIPPAGE_PCT)))
                if qty > 0:
                    fill = price * (1 + config.SLIPPAGE_PCT)
                    cost = qty * fill + config.BROKERAGE_PER_ORDER
                    if cost <= cash:
                        cash -= cost
                        positions[symbol] = {"qty": qty, "avg": fill}
                        trades.append({"type": "BUY", "symbol": symbol, "date": str(d), "price": round(fill, 2), "qty": qty})
                continue

            if pos is None:
                continue

            pnl_pct = (price - pos["avg"]) / pos["avg"]
            exit_type = None
            if signal == "SELL":
                exit_type = "SELL"
            elif pnl_pct <= -config.STOP_LOSS_PCT:
                exit_type = "STOP_LOSS"
            elif pnl_pct >= config.TAKE_PROFIT_PCT:
                exit_type = "TAKE_PROFIT"

            if exit_type:
                fill = price * (1 - config.SLIPPAGE_PCT)
                proceeds = pos["qty"] * fill - config.BROKERAGE_PER_ORDER
                pnl = (fill - pos["avg"]) * pos["qty"]
                cash += proceeds
                trades.append({
                    "type": exit_type,
                    "symbol": symbol,
                    "date": str(d),
                    "price": round(fill, 2),
                    "qty": pos["qty"],
                    "pnl": round(pnl, 2),
                })
                del positions[symbol]

    # Mark-to-market remaining positions at final available close
    mtm_value = 0.0
    for symbol, pos in positions.items():
        df = prepped[symbol]
        last_close = float(df["Close"].iloc[-1])
        mtm_value += pos["qty"] * last_close

    final_equity = cash + mtm_value
    realized_trades = [t for t in trades if t["type"] != "BUY"]
    winners = [t for t in realized_trades if t.get("pnl", 0) > 0]
    losers = [t for t in realized_trades if t.get("pnl", 0) < 0]

    return {
        "initial_capital": round(initial_capital, 2),
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(((final_equity - initial_capital) / initial_capital) * 100, 2),
        "total_trades": len(realized_trades),
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "win_rate": round((len(winners) / len(realized_trades) * 100), 1) if realized_trades else 0.0,
        "total_pnl": round(sum(t.get("pnl", 0) for t in realized_trades), 2),
        "open_positions": len(positions),
        "trades": trades,
    }


def print_backtest_report(results: list[dict]):
    """Print a formatted backtest report for multiple symbols."""
    table_data = []
    total_pnl = 0

    for r in results:
        if "error" in r:
            table_data.append([r["symbol"], "ERROR", "-", "-", "-", "-"])
            continue
        total_pnl += r["total_pnl"]
        table_data.append([
            r["symbol"],
            f"Rs.{r['final_equity']:.2f}",
            f"{r['total_return_pct']}%",
            r["total_trades"],
            f"{r['win_rate']}%",
            f"Rs.{r['total_pnl']:.2f}",
        ])

    print("\n" + "=" * 70)
    print("BACKTEST REPORT")
    print("=" * 70)
    print(tabulate(
        table_data,
        headers=["Symbol", "Final Equity", "Return", "Trades", "Win Rate", "P&L"],
        tablefmt="grid",
    ))
    print(f"\nTotal P&L across all symbols: Rs.{total_pnl:.2f}")
    print("=" * 70)
