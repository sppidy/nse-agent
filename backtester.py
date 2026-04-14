"""Backtesting engine — test strategies on historical data."""

from decimal import Decimal
import pandas as pd
from tabulate import tabulate

import config
from strategy import generate_signals
from paper_trader import D


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

    cash = D(initial_capital)
    position = 0
    avg_price = D(0)
    slippage = D(config.SLIPPAGE_PCT)
    brokerage = D(config.BROKERAGE_PER_ORDER)
    max_pos_pct = D(config.MAX_POSITION_SIZE_PCT)
    stop_loss = D(config.STOP_LOSS_PCT)
    take_profit = D(config.TAKE_PROFIT_PCT)
    trades = []
    equity_curve = []

    for i, row in df.iterrows():
        price = D(row["Close"])
        signal = row["signal"]

        # Track equity
        equity = cash + D(position) * price
        equity_curve.append({"date": i, "equity": float(equity)})

        if signal == "BUY" and position == 0:
            # Buy with position sizing
            max_spend = cash * max_pos_pct
            fill_price = price * (1 + slippage)
            qty = int(max_spend / fill_price) if fill_price > 0 else 0
            if qty > 0:
                cost = D(qty) * fill_price + brokerage
                cash -= cost
                position = qty
                avg_price = fill_price
                trades.append({
                    "type": "BUY",
                    "date": str(i),
                    "price": float(round(fill_price, 2)),
                    "qty": qty,
                })

        elif signal == "SELL" and position > 0:
            fill = price * (1 - slippage)
            proceeds = D(position) * fill - brokerage
            pnl = (fill - avg_price) * D(position)
            cash += proceeds
            trades.append({
                "type": "SELL",
                "date": str(i),
                "price": float(round(fill, 2)),
                "qty": position,
                "pnl": float(round(pnl, 2)),
            })
            position = 0

        # Stop loss / take profit
        elif position > 0 and avg_price > 0:
            pnl_pct = (price - avg_price) / avg_price
            if pnl_pct <= -stop_loss:
                fill = price * (1 - slippage)
                proceeds = D(position) * fill - brokerage
                pnl = (fill - avg_price) * D(position)
                cash += proceeds
                trades.append({
                    "type": "STOP_LOSS",
                    "date": str(i),
                    "price": float(round(fill, 2)),
                    "qty": position,
                    "pnl": float(round(pnl, 2)),
                })
                position = 0
            elif pnl_pct >= take_profit:
                fill = price * (1 - slippage)
                proceeds = D(position) * fill - brokerage
                pnl = (fill - avg_price) * D(position)
                cash += proceeds
                trades.append({
                    "type": "TAKE_PROFIT",
                    "date": str(i),
                    "price": float(round(fill, 2)),
                    "qty": position,
                    "pnl": float(round(pnl, 2)),
                })
                position = 0

    # Close any remaining position at last price
    final_price = D(df.iloc[-1]["Close"])
    if position > 0:
        fill = final_price * (1 - slippage)
        proceeds = D(position) * fill
        pnl = (fill - avg_price) * D(position)
        cash += proceeds
        trades.append({
            "type": "CLOSE",
            "date": str(df.index[-1]),
            "price": float(round(fill, 2)),
            "qty": position,
            "pnl": float(round(pnl, 2)),
        })
        position = 0

    initial_d = D(initial_capital)
    final_equity = cash
    total_return = float((final_equity - initial_d) / initial_d * 100) if initial_d > 0 else 0
    winning = [t for t in trades if t.get("pnl", 0) > 0]
    losing = [t for t in trades if t.get("pnl", 0) < 0]
    sell_trades = [t for t in trades if t["type"] != "BUY"]

    return {
        "symbol": symbol,
        "initial_capital": float(initial_d),
        "final_equity": float(round(final_equity, 2)),
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

    cash = D(initial_capital)
    slippage = D(config.SLIPPAGE_PCT)
    brokerage = D(config.BROKERAGE_PER_ORDER)
    max_pos_pct = D(config.MAX_POSITION_SIZE_PCT)
    stop_loss = D(config.STOP_LOSS_PCT)
    take_profit = D(config.TAKE_PROFIT_PCT)
    positions: dict[str, dict] = {}
    trades = []

    for d in sorted(all_dates):
        for symbol, df in prepped.items():
            if d not in df.index:
                continue
            row = df.loc[d]
            price = D(row["Close"])
            signal = row["signal"]
            pos = positions.get(symbol)

            if signal == "BUY" and pos is None:
                max_spend = cash * max_pos_pct
                fill_price = price * (1 + slippage)
                qty = int(max_spend / fill_price) if fill_price > 0 else 0
                if qty > 0:
                    cost = D(qty) * fill_price + brokerage
                    if cost <= cash:
                        cash -= cost
                        positions[symbol] = {"qty": qty, "avg": fill_price}
                        trades.append({"type": "BUY", "symbol": symbol, "date": str(d), "price": float(round(fill_price, 2)), "qty": qty})
                continue

            if pos is None:
                continue

            pos_avg = pos["avg"]
            pnl_pct = (price - pos_avg) / pos_avg if pos_avg > 0 else D(0)
            exit_type = None
            if signal == "SELL":
                exit_type = "SELL"
            elif pnl_pct <= -stop_loss:
                exit_type = "STOP_LOSS"
            elif pnl_pct >= take_profit:
                exit_type = "TAKE_PROFIT"

            if exit_type:
                fill = price * (1 - slippage)
                proceeds = D(pos["qty"]) * fill - brokerage
                pnl = (fill - pos_avg) * D(pos["qty"])
                cash += proceeds
                trades.append({
                    "type": exit_type,
                    "symbol": symbol,
                    "date": str(d),
                    "price": float(round(fill, 2)),
                    "qty": pos["qty"],
                    "pnl": float(round(pnl, 2)),
                })
                del positions[symbol]

    # Mark-to-market remaining positions at final available close
    mtm_value = D(0)
    for symbol, pos in positions.items():
        df = prepped[symbol]
        last_close = D(df["Close"].iloc[-1])
        mtm_value += D(pos["qty"]) * last_close

    initial_d = D(initial_capital)
    final_equity = cash + mtm_value
    realized_trades = [t for t in trades if t["type"] != "BUY"]
    winners = [t for t in realized_trades if t.get("pnl", 0) > 0]
    losers = [t for t in realized_trades if t.get("pnl", 0) < 0]
    total_return = float((final_equity - initial_d) / initial_d * 100) if initial_d > 0 else 0

    return {
        "initial_capital": float(round(initial_d, 2)),
        "final_equity": float(round(final_equity, 2)),
        "total_return_pct": round(total_return, 2),
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
