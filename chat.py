"""Interactive chat interface for the AI Trading Agent — Rich UI edition."""

import os
import sys
import json
import time
from datetime import datetime

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich.columns import Columns
from rich.live import Live
from rich.spinner import Spinner
from rich.layout import Layout
from rich.rule import Rule
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter

import config
from paper_trader import PaperTrader, Portfolio
from data_fetcher import get_watchlist_prices, get_historical_data, get_live_price
from strategy import add_indicators
from learner import get_learning_context, get_performance_stats
from news_sentiment import fetch_all_news, format_news_for_ai
from logger import logger

load_dotenv()

console = Console()

def _call_ai(prompt: str) -> str:
    """Call Groq first, Gemini as fallback (sync wrapper)."""
    # Try Groq first
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            models = [
                "llama-3.3-70b-versatile",
                "qwen/qwen3-32b",
                "llama-3.1-8b-instant",
            ]
            for model in models:
                for attempt in range(3):
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.4,
                            max_tokens=1024,
                        )
                        return response.choices[0].message.content.strip()
                    except Exception as e:
                        err = str(e)
                        if "429" in err or "rate_limit" in err.lower():
                            if attempt < 2:
                                time.sleep(3 * (attempt + 1))
                            else:
                                break
                        elif "404" in err:
                            break
                        else:
                            raise
        except Exception as e:
            logger.warning(f"Groq failed, falling back to Gemini: {e}")

    # Fallback to Gemini
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and gemini_key != "your_api_key_here":
        from google import genai
        client = genai.Client(api_key=gemini_key)
        models = [
            "gemma-4-31b-it",
            "gemma-4-26b-a4b-it",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
        ]
        for model in models:
            for attempt in range(3):
                try:
                    response = client.models.generate_content(model=model, contents=prompt)
                    return response.text.strip()
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        if attempt < 2:
                            time.sleep(5 * (attempt + 2))
                        else:
                            break
                    elif "404" in err or "NOT_FOUND" in err:
                        break
                    else:
                        raise

    return "All AI models are rate-limited right now. Try again in a minute."


def _get_portfolio_data() -> dict:
    """Get structured portfolio data."""
    trader = PaperTrader()
    prices = get_watchlist_prices()
    summary = trader.get_summary(prices)
    return {
        "trader": trader,
        "prices": prices,
        "summary": summary,
    }


def _get_portfolio_text(trader, prices, summary) -> str:
    """Build compact portfolio text for AI context."""
    text = f"Cash:Rs.{summary['cash']:.0f},Value:Rs.{summary['total_value']:.0f},Ret:{summary['total_return_pct']:+.2f}%,P&L:Rs.{summary['realized_pnl']:.1f}\n"
    if trader.portfolio.positions:
        for sym, pos in trader.portfolio.positions.items():
            current = prices.get(sym, pos.avg_price)
            pnl_pct = pos.pnl_pct(current)
            text += f"POS:{sym.replace('.NS','')}|{pos.quantity}x|{pos.avg_price:.1f}->{current:.1f}|{pnl_pct:+.1f}%\n"
    if trader.portfolio.trade_log:
        for t in trader.portfolio.trade_log[-3:]:
            text += f"CLOSED:{t['symbol'].replace('.NS','')}|Rs.{t['pnl']:.1f}|{t['pnl_pct']:+.1f}%\n"
    return text


def _get_market_snapshot() -> str:
    """Build compact market data snapshot."""
    lines = []
    for symbol in config.WATCHLIST:
        try:
            df = get_historical_data(symbol, period="30d", interval="1d")
            if df.empty:
                continue
            df = add_indicators(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            chg = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100 if prev["Close"] != 0 else 0
            ema = "B" if latest["ema_short"] > latest["ema_long"] else "X"
            lines.append(f"{symbol.replace('.NS','')}|{latest['Close']:.1f}|{chg:+.1f}%|RSI:{latest['rsi']:.0f}|EMA:{ema}")
        except Exception:
            logger.warning(f"Market snapshot skipped for {symbol}: data fetch/indicator error")
    return "\n".join(lines) if lines else "No data."


def render_header():
    """Render the app header."""
    now = datetime.now().strftime("%d-%b-%Y %I:%M %p")
    title = Text("AI TRADING AGENT", style="bold cyan")
    subtitle = Text(f" {now} IST", style="dim")
    header_text = Text.assemble(title, subtitle)
    console.print(Panel(header_text, border_style="cyan", padding=(0, 2)))


def render_portfolio(trader, prices, summary):
    """Render portfolio panel with rich tables."""
    # Main stats
    ret_style = "green" if summary["total_return_pct"] >= 0 else "red"
    pnl_style = "green" if summary["realized_pnl"] >= 0 else "red"

    stats_table = Table(show_header=False, box=None, padding=(0, 2))
    stats_table.add_column("label", style="dim")
    stats_table.add_column("value", justify="right")
    stats_table.add_row("Cash", f"Rs.{summary['cash']:.2f}")
    stats_table.add_row("Positions", f"Rs.{summary['positions_value']:.2f}")
    stats_table.add_row("Total Value", f"[bold]Rs.{summary['total_value']:.2f}[/bold]")
    stats_table.add_row("Return", f"[{ret_style}]{summary['total_return_pct']:+.2f}%[/{ret_style}]")
    stats_table.add_row("Realized P&L", f"[{pnl_style}]Rs.{summary['realized_pnl']:.2f}[/{pnl_style}]")

    # Last trade time
    if trader.portfolio.orders:
        last_ts = trader.portfolio.orders[-1].get("timestamp", "")
        if last_ts:
            try:
                dt = datetime.fromisoformat(last_ts)
                stats_table.add_row("Last Trade", dt.strftime("%I:%M %p %d-%b"))
            except ValueError:
                pass

    console.print(Panel(stats_table, title="[bold]Portfolio[/bold]", border_style="blue", width=42))

    # Open positions table
    if trader.portfolio.positions:
        pos_table = Table(title="Open Positions", border_style="blue", show_lines=False)
        pos_table.add_column("Stock", style="cyan")
        pos_table.add_column("Qty", justify="right")
        pos_table.add_column("Entry", justify="right")
        pos_table.add_column("Current", justify="right")
        pos_table.add_column("P&L", justify="right")

        for sym, pos in trader.portfolio.positions.items():
            current = prices.get(sym, pos.avg_price)
            pnl = pos.pnl(current)
            pnl_pct = pos.pnl_pct(current)
            style = "green" if pnl >= 0 else "red"
            pnl_text = f"[{style}]Rs.{pnl:+.2f} ({pnl_pct:+.1f}%)[/{style}]"
            pos_table.add_row(
                sym.replace(".NS", ""),
                str(pos.quantity),
                f"Rs.{pos.avg_price:.2f}",
                f"Rs.{current:.2f}",
                pnl_text,
            )
        console.print(pos_table)

    # Recent trades
    if trader.portfolio.trade_log:
        trade_table = Table(title="Recent Trades", border_style="green", show_lines=False)
        trade_table.add_column("Stock", style="cyan")
        trade_table.add_column("Entry", justify="right")
        trade_table.add_column("Exit", justify="right")
        trade_table.add_column("P&L", justify="right")
        trade_table.add_column("Time", style="dim")

        for t in trader.portfolio.trade_log[-5:]:
            pnl = t["pnl"]
            style = "green" if pnl >= 0 else "red"
            trade_time = ""
            if t.get("timestamp"):
                try:
                    dt = datetime.fromisoformat(t["timestamp"])
                    trade_time = dt.strftime("%I:%M %p %d-%b")
                except ValueError:
                    pass
            trade_table.add_row(
                t["symbol"].replace(".NS", ""),
                f"Rs.{t['entry_price']:.2f}",
                f"Rs.{t['exit_price']:.2f}",
                f"[{style}]Rs.{pnl:+.2f} ({t['pnl_pct']:+.1f}%)[/{style}]",
                trade_time,
            )
        console.print(trade_table)


def render_help():
    """Render help panel."""
    help_table = Table(show_header=False, box=None, padding=(0, 1))
    help_table.add_column("cmd", style="bold yellow")
    help_table.add_column("desc", style="dim")
    help_table.add_row("buy SYMBOL [qty]", "Buy a stock  (e.g. buy SBIN 10)")
    help_table.add_row("sell SYMBOL [qty]", "Sell a stock (e.g. sell WIPRO)")
    help_table.add_row("status", "Show portfolio dashboard")
    help_table.add_row("refresh", "Refresh market data")
    help_table.add_row("help", "Show this help")
    help_table.add_row("quit", "Exit chat")
    console.print(Panel(help_table, title="[bold]Commands[/bold]", border_style="yellow", width=55))
    console.print("[dim]Or just type naturally — ask about stocks, strategies, risks, anything![/dim]\n")


def _execute_command(user_input: str, trader: PaperTrader) -> str | None:
    """Handle direct action commands."""
    lower = user_input.strip().lower()

    if lower.startswith("buy "):
        parts = lower.split()
        if len(parts) >= 2:
            symbol = parts[1].upper()
            if not symbol.endswith(".NS"):
                symbol += ".NS"
            qty = int(parts[2]) if len(parts) >= 3 else None
            price = get_live_price(symbol)
            if price <= 0:
                return f"[red]Could not fetch price for {symbol}.[/red]"
            order = trader.buy(symbol, price, quantity=qty)
            if order:
                return f"[green]Bought {order.quantity}x {symbol} @ Rs.{order.fill_price():.2f}[/green]"
            return f"[red]Could not buy {symbol}. Check funds or position limits.[/red]"

    if lower.startswith("sell "):
        parts = lower.split()
        if len(parts) >= 2:
            symbol = parts[1].upper()
            if not symbol.endswith(".NS"):
                symbol += ".NS"
            qty = int(parts[2]) if len(parts) >= 3 else None
            price = get_live_price(symbol)
            if price <= 0:
                return f"[red]Could not fetch price for {symbol}.[/red]"
            order = trader.sell(symbol, price, quantity=qty)
            if order:
                fill = order.fill_price()
                return f"[green]Sold {order.quantity}x {symbol} @ Rs.{fill:.2f}[/green]"
            return f"[red]Could not sell {symbol}. Check if you hold it.[/red]"

    return None


def build_system_prompt() -> str:
    return f"""NSE paper trading AI assistant. Date:{datetime.now().strftime('%d-%b %I:%M%p')} IST. Market:9:15AM-3:30PM Mon-Fri.
Config: Rs.{config.INITIAL_CAPITAL} capital, {config.MAX_POSITION_SIZE_PCT*100}%/pos, {config.MAX_OPEN_POSITIONS} max pos, SL:{config.STOP_LOSS_PCT*100}%, TP:{config.TAKE_PROFIT_PCT*100}%
Watchlist: {','.join(s.replace('.NS','') for s in config.WATCHLIST)}

System runs: Technicals(RSI14,EMA9/21)+AI(Gemini)+News(GoogleNews,Yahoo,Twitter/X,ET,Moneycontrol)+ML(GradientBoosting,observing<55%acc)+TradeHistory. Confidence>60% to trade.

Rules: Be concise(<150 words). Use Rs. Use provided data only. For buy/sell tell user: buy SYMBOL [qty]. Mention risks. Use markdown. You DO use news sentiment from 5 sources actively."""


def chat():
    """Main interactive chat loop with Rich UI."""
    console.clear()
    render_header()

    # Command autocomplete
    commands = ["buy", "sell", "status", "refresh", "help", "quit", "exit"]
    stock_names = [s.replace(".NS", "").lower() for s in config.WATCHLIST]
    completer = WordCompleter(commands + stock_names + config.WATCHLIST, ignore_case=True)
    history = InMemoryHistory()

    # Initialize

    with console.status("[cyan]Loading portfolio, market & news data...", spinner="dots"):
        pdata = _get_portfolio_data()
        market_data = _get_market_snapshot()
        learning_data = get_learning_context()
        stats = get_performance_stats()
        # Load latest news sentiment
        try:
            news_data = fetch_all_news()
            news_text = format_news_for_ai(news_data)
        except Exception:
            logger.warning("News fetch failed during startup; using fallback text.")
            news_text = "News data unavailable."

    trader = pdata["trader"]
    prices = pdata["prices"]
    summary = pdata["summary"]

    # Show initial dashboard
    render_portfolio(trader, prices, summary)
    console.print()
    render_help()

    stats_text = ""
    if stats and stats.get("total_trades", 0) > 0:
        stats_text = (
            f"\nPERFORMANCE: Win Rate {stats.get('win_rate', 0):.0f}%, "
            f"Avg Win Rs.{stats.get('avg_win_pct', 0):.2f}%, "
            f"Avg Loss Rs.{stats.get('avg_loss_pct', 0):.2f}%"
        )

    portfolio_text = _get_portfolio_text(trader, prices, summary)
    system_prompt = build_system_prompt()
    conversation = []

    def _get_input():
        try:
            return pt_prompt(
                "You > ",
                completer=completer,
                history=history,
                auto_suggest=AutoSuggestFromHistory(),
            ).strip()
        except Exception:
            # Fallback for non-interactive terminals
            return input("You > ").strip()

    while True:
        try:
            user_input = _get_input()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[cyan]Goodbye! Happy trading.[/cyan]")
            break

        if not user_input:
            continue

        lower = user_input.lower()

        if lower in ("quit", "exit", "bye", "q"):
            console.print("[cyan]Goodbye! Happy trading.[/cyan]")
            break

        if lower == "help":
            render_help()
            continue

        if lower == "status":
            with console.status("[cyan]Refreshing...", spinner="dots"):
                pdata = _get_portfolio_data()
                trader, prices, summary = pdata["trader"], pdata["prices"], pdata["summary"]
                portfolio_text = _get_portfolio_text(trader, prices, summary)
            render_portfolio(trader, prices, summary)
            console.print()
            continue

        if lower == "refresh":
            with console.status("[cyan]Refreshing all data...", spinner="dots"):
                pdata = _get_portfolio_data()
                trader, prices, summary = pdata["trader"], pdata["prices"], pdata["summary"]
                portfolio_text = _get_portfolio_text(trader, prices, summary)
                market_data = _get_market_snapshot()
                learning_data = get_learning_context()
                try:
                    news_data = fetch_all_news()
                    news_text = format_news_for_ai(news_data)
                except Exception:
                    logger.warning("News refresh failed; keeping previously fetched news context.")
            console.print("[green]Data refreshed![/green]\n")
            continue

        # Handle buy/sell
        cmd_result = _execute_command(user_input, trader)
        if cmd_result:
            console.print(f"\n  {cmd_result}\n")
            pdata = _get_portfolio_data()
            trader, prices, summary = pdata["trader"], pdata["prices"], pdata["summary"]
            portfolio_text = _get_portfolio_text(trader, prices, summary)
            continue

        # AI chat
        conversation.append({"role": "user", "content": user_input})

        recent_convo = conversation[-10:]
        convo_text = ""
        for msg in recent_convo[:-1]:
            prefix = "U" if msg["role"] == "user" else "A"
            # Truncate old messages to save tokens
            content = msg["content"][:200]
            convo_text += f"{prefix}:{content}\n"

        ai_prompt = f"""{system_prompt}

PORTFOLIO DATA:
{portfolio_text}
{stats_text}

MARKET DATA:
{market_data}

{news_text}

TRADE HISTORY:
{learning_data}

CONVERSATION:
{convo_text}

User: {user_input}

Respond helpfully and concisely:"""

        with console.status("[cyan]Thinking...", spinner="dots"):
            response = _call_ai(ai_prompt)

        console.print()
        console.print(Panel(
            Markdown(response),
            title="[bold cyan]Agent[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        ))
        console.print()

        conversation.append({"role": "assistant", "content": response})

        # Refresh portfolio every 5 exchanges
        if len(conversation) % 10 == 0:
            pdata = _get_portfolio_data()
            trader, prices, summary = pdata["trader"], pdata["prices"], pdata["summary"]
            portfolio_text = _get_portfolio_text(trader, prices, summary)


if __name__ == "__main__":
    chat()
