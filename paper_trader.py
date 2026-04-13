"""Paper trading engine — simulates order execution and tracks portfolio."""

import os
import random
import math
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum

import config
from logger import logger
from market_calendar import is_market_trading_day
from persistence import read_json, write_json_atomic


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: str
    status: OrderStatus = OrderStatus.FILLED
    order_id: str = ""
    slippage: float = 0.0
    brokerage: float = 0.0

    def fill_price(self) -> float:
        if self.side == OrderSide.BUY:
            return self.price * (1 + self.slippage)
        return self.price * (1 - self.slippage)

    def total_cost(self) -> float:
        return self.quantity * self.fill_price() + self.brokerage


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float
    entry_time: str
    highest_price: float = 0.0
    signal_confidence: float = 0.0
    ai_stop_loss: float | None = None
    ai_target: float | None = None
    dynamic_stop_loss_pct: float = 0.0
    dynamic_take_profit_pct: float = 0.0

    def __post_init__(self):
        if self.highest_price == 0.0:
            self.highest_price = self.avg_price
        if self.dynamic_stop_loss_pct <= 0:
            self.dynamic_stop_loss_pct = config.STOP_LOSS_PCT
        if self.dynamic_take_profit_pct <= 0:
            self.dynamic_take_profit_pct = config.TAKE_PROFIT_PCT

    def current_value(self, current_price: float) -> float:
        return self.quantity * current_price

    def pnl(self, current_price: float) -> float:
        return (current_price - self.avg_price) * self.quantity

    def pnl_pct(self, current_price: float) -> float:
        if self.avg_price == 0:
            return 0
        return ((current_price - self.avg_price) / self.avg_price) * 100


@dataclass
class Portfolio:
    cash: float = config.INITIAL_CAPITAL
    positions: dict[str, Position] = field(default_factory=dict)
    orders: list[dict] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)
    total_realized_pnl: float = 0.0

    @staticmethod
    def _safe_price(value: float | None, fallback: float) -> float:
        if value is None:
            return fallback
        try:
            price = float(value)
        except (TypeError, ValueError):
            return fallback
        if not math.isfinite(price) or price <= 0:
            return fallback
        return price

    def total_value(self, prices: dict[str, float]) -> float:
        positions_value = sum(
            pos.current_value(self._safe_price(prices.get(sym), pos.avg_price))
            for sym, pos in self.positions.items()
        )
        return self.cash + positions_value

    @staticmethod
    def _resolve_path(filepath: str) -> str:
        if os.path.isabs(filepath):
            return filepath
        return os.path.join(config.PROJECT_DIR, filepath)

    def save(self, filepath: str = "portfolio.json"):
        filepath = self._resolve_path(filepath)
        data = {
            "cash": self.cash,
            "positions": {
                sym: asdict(pos) for sym, pos in self.positions.items()
            },
            "orders": self.orders,
            "trade_log": self.trade_log,
            "total_realized_pnl": self.total_realized_pnl,
        }
        write_json_atomic(filepath, data)

    @classmethod
    def load(cls, filepath: str = "portfolio.json") -> "Portfolio":
        filepath = cls._resolve_path(filepath)
        if not os.path.exists(filepath):
            return cls()
        data = read_json(filepath, default={})
        portfolio = cls(
            cash=data.get("cash", config.INITIAL_CAPITAL),
            total_realized_pnl=data.get("total_realized_pnl", 0),
        )
        for sym, pos_data in data.get("positions", {}).items():
            portfolio.positions[sym] = Position(**pos_data)
        portfolio.orders = data.get("orders", [])
        portfolio.trade_log = data.get("trade_log", [])
        return portfolio


class PaperTrader:
    def __init__(self, portfolio: Portfolio | None = None, filepath: str = "portfolio.json"):
        self.filepath = Portfolio._resolve_path(filepath)
        self._disk_sync_enabled = portfolio is None
        self.portfolio = portfolio or Portfolio.load(self.filepath)
        self._order_counter = len(self.portfolio.orders)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(value, high))

    def refresh_portfolio(self) -> None:
        if not self._disk_sync_enabled:
            return
        latest = Portfolio.load(self.filepath)
        self.portfolio = latest
        self._order_counter = max(self._order_counter, len(self.portfolio.orders))

    def _capital_utilization_floor_pct(self, max_position_size_pct: float) -> float:
        initial_capital = max(config.INITIAL_CAPITAL, 1.0)
        cash_ratio = self.portfolio.cash / initial_capital
        deploy_gap = max(0.0, cash_ratio - (1.0 - config.CAPITAL_DEPLOYMENT_TARGET_PCT))
        slot_budget = deploy_gap / max(config.MAX_OPEN_POSITIONS, 1)
        floor_pct = max(config.CAPITAL_UTILIZATION_MIN_BET_PCT, slot_budget)
        return self._clamp(floor_pct, 0.0, max_position_size_pct)

    def _dynamic_risk_levels(
        self,
        entry_price: float,
        confidence: float,
        ai_signal: dict | None,
    ) -> tuple[float, float, float | None, float | None]:
        safe_conf = self._clamp(float(confidence or 0.0), 0.0, 1.0)
        stop_pct = config.STOP_LOSS_PCT
        take_pct = config.TAKE_PROFIT_PCT

        ai_stop = None
        ai_target = None
        if ai_signal:
            ai_stop_raw = ai_signal.get("stop_loss")
            ai_target_raw = ai_signal.get("target")
            try:
                ai_stop_val = float(ai_stop_raw) if ai_stop_raw is not None else None
            except (TypeError, ValueError):
                ai_stop_val = None
            try:
                ai_target_val = float(ai_target_raw) if ai_target_raw is not None else None
            except (TypeError, ValueError):
                ai_target_val = None

            if ai_stop_val is not None and 0 < ai_stop_val < entry_price:
                ai_stop = ai_stop_val
                stop_pct = (stop_pct + ((entry_price - ai_stop_val) / entry_price)) / 2.0
            if ai_target_val is not None and ai_target_val > entry_price:
                ai_target = ai_target_val
                take_pct = (take_pct + ((ai_target_val - entry_price) / entry_price)) / 2.0

        conf_shift = (0.5 - safe_conf) * config.TRAILING_CONFIDENCE_SCALE
        stop_pct *= (1.0 + conf_shift)
        take_pct *= (1.0 - conf_shift)

        stop_pct = self._clamp(stop_pct, config.MIN_STOP_LOSS_PCT, config.MAX_STOP_LOSS_PCT)
        take_pct = self._clamp(take_pct, config.MIN_TAKE_PROFIT_PCT, config.MAX_TAKE_PROFIT_PCT)
        return stop_pct, take_pct, ai_stop, ai_target

    def buy(
        self,
        symbol: str,
        price: float,
        quantity: int | None = None,
        confidence: float = 0.0,
        max_position_size_pct: float | None = None,
        ai_signal: dict | None = None,
    ) -> Order | None:
        """Place a simulated buy order."""
        self.refresh_portfolio()
        if not is_market_trading_day():
            logger.warning(f"Cannot buy {symbol}: non-trading day (weekend/holiday)")
            return None

        if max_position_size_pct is None:
            max_position_size_pct = config.MAX_POSITION_SIZE_PCT

        # Simulate dynamic slippage (liquidity constraints)
        slippage = config.SLIPPAGE_PCT * random.uniform(0.5, 2.0)
        fill_price = price * (1 + slippage)
        brokerage = config.BROKERAGE_PER_ORDER

        # Auto-calculate quantity if not specified
        if quantity is None:
            # Phase 2: Kelly Criterion Dynamic Position Sizing
            if confidence > 0:
                W = confidence
                # Reward/Risk Ratio (e.g., 3% / 2% = 1.5)
                R = config.TAKE_PROFIT_PCT / config.STOP_LOSS_PCT
                # Kelly fraction: f = W - ((1 - W) / R)
                kelly_fraction = W - ((1 - W) / R)
                # Half-Kelly for safety
                half_kelly = kelly_fraction / 2.0
                utilization_floor = self._capital_utilization_floor_pct(max_position_size_pct)
                # Cap the maximum bet at the config limit and ensure it is not negative.
                optimal_bet_pct = max(utilization_floor, min(half_kelly, max_position_size_pct))
                max_spend = self.portfolio.cash * optimal_bet_pct
            else:
                # Fallback to static if no confidence is provided
                max_spend = self.portfolio.cash * max_position_size_pct
                
            quantity = int(max_spend / fill_price)

        if quantity <= 0:
            logger.warning(
                f"Cannot buy {symbol}: insufficient funds "
                f"(need Rs.{fill_price:.2f}, have Rs.{self.portfolio.cash:.2f})"
            )
            return None

        total_cost = quantity * fill_price + brokerage
        if total_cost > self.portfolio.cash:
            # Reduce quantity to fit
            quantity = int((self.portfolio.cash - brokerage) / fill_price)
            if quantity <= 0:
                logger.warning(f"Cannot buy {symbol}: insufficient funds")
                return None
            total_cost = quantity * fill_price + brokerage

        # Check max positions
        if symbol not in self.portfolio.positions and len(self.portfolio.positions) >= config.MAX_OPEN_POSITIONS:
            logger.warning(f"Cannot buy {symbol}: max {config.MAX_OPEN_POSITIONS} positions reached")
            return None

        # Execute
        self._order_counter += 1
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            price=price,
            timestamp=datetime.now().isoformat(),
            order_id=f"ORD-{self._order_counter:04d}",
            slippage=slippage,
            brokerage=brokerage,
        )

        self.portfolio.cash -= total_cost
        dynamic_stop, dynamic_take, ai_stop, ai_target = self._dynamic_risk_levels(fill_price, confidence, ai_signal)

        if symbol in self.portfolio.positions:
            pos = self.portfolio.positions[symbol]
            total_qty = pos.quantity + quantity
            pos.avg_price = (pos.avg_price * pos.quantity + fill_price * quantity) / total_qty
            pos.quantity = total_qty
            pos.signal_confidence = max(pos.signal_confidence, confidence)
            pos.dynamic_stop_loss_pct = dynamic_stop
            pos.dynamic_take_profit_pct = dynamic_take
            if ai_stop is not None:
                pos.ai_stop_loss = ai_stop
            if ai_target is not None:
                pos.ai_target = ai_target
        else:
            self.portfolio.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=fill_price,
                entry_time=order.timestamp,
                signal_confidence=confidence,
                ai_stop_loss=ai_stop,
                ai_target=ai_target,
                dynamic_stop_loss_pct=dynamic_stop,
                dynamic_take_profit_pct=dynamic_take,
            )

        self.portfolio.orders.append(asdict(order))
        if self._disk_sync_enabled:
            self.portfolio.save(self.filepath)
        logger.info(f"BUY {quantity}x {symbol} @ Rs.{fill_price:.2f} = Rs.{total_cost:.2f}")
        return order

    def sell(self, symbol: str, price: float, quantity: int | None = None) -> Order | None:
        """Place a simulated sell order."""
        self.refresh_portfolio()
        if not is_market_trading_day():
            logger.warning(f"Cannot sell {symbol}: non-trading day (weekend/holiday)")
            return None

        if symbol not in self.portfolio.positions:
            logger.warning(f"Cannot sell {symbol}: no position held")
            return None

        pos = self.portfolio.positions[symbol]
        if quantity is None:
            quantity = pos.quantity

        quantity = min(quantity, pos.quantity)
        if quantity <= 0:
            return None

        # Simulate dynamic slippage (liquidity constraints)
        slippage = config.SLIPPAGE_PCT * random.uniform(0.5, 2.0)
        fill_price = price * (1 - slippage)
        brokerage = config.BROKERAGE_PER_ORDER

        self._order_counter += 1
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            price=price,
            timestamp=datetime.now().isoformat(),
            order_id=f"ORD-{self._order_counter:04d}",
            slippage=slippage,
            brokerage=brokerage,
        )

        proceeds = quantity * fill_price - brokerage
        self.portfolio.cash += proceeds

        # Calculate realized P&L
        pnl = (fill_price - pos.avg_price) * quantity
        self.portfolio.total_realized_pnl += pnl

        self.portfolio.trade_log.append({
            "symbol": symbol,
            "side": "SELL",
            "quantity": quantity,
            "entry_price": pos.avg_price,
            "exit_price": fill_price,
            "pnl": round(pnl, 2),
            "pnl_pct": round(((fill_price - pos.avg_price) / pos.avg_price) * 100, 2),
            "timestamp": order.timestamp,
        })

        if quantity >= pos.quantity:
            del self.portfolio.positions[symbol]
        else:
            pos.quantity -= quantity

        self.portfolio.orders.append(asdict(order))
        if self._disk_sync_enabled:
            self.portfolio.save(self.filepath)
        pnl_str = f"+Rs.{pnl:.2f}" if pnl >= 0 else f"-Rs.{abs(pnl):.2f}"
        logger.info(f"SELL {quantity}x {symbol} @ Rs.{fill_price:.2f} | P&L: {pnl_str}")
        return order

    def check_stop_loss_take_profit(self, prices: dict[str, float]) -> list[Order]:
        """Check and execute trailing stop-loss / take-profit for all positions."""
        self.refresh_portfolio()
        if not is_market_trading_day():
            logger.info("Skipping stop-loss/take-profit checks: non-trading day (weekend/holiday)")
            return []

        orders = []
        for symbol, pos in list(self.portfolio.positions.items()):
            current = prices.get(symbol)
            if current is None:
                continue
                
            # Update highest price seen for trailing stop
            pos.highest_price = max(pos.highest_price, current)
            if self._disk_sync_enabled:
                self.portfolio.save(self.filepath)

            pnl_pct = pos.pnl_pct(current) / 100

            base_stop = pos.dynamic_stop_loss_pct if config.DYNAMIC_TRAILING_ENABLED else config.STOP_LOSS_PCT
            base_take = pos.dynamic_take_profit_pct if config.DYNAMIC_TRAILING_ENABLED else config.TAKE_PROFIT_PCT
            runup_pct = max(0.0, (pos.highest_price - pos.avg_price) / pos.avg_price) if pos.avg_price > 0 else 0.0
            tightened_stop = max(config.MIN_STOP_LOSS_PCT, base_stop - (runup_pct * config.TRAILING_PROFIT_LOCK_SCALE))
            trailing_loss_pct = (current - pos.highest_price) / pos.highest_price

            if trailing_loss_pct <= -tightened_stop:
                logger.info(f"TRAILING STOP LOSS triggered for {symbol} ({trailing_loss_pct*100:.1f}% from peak of Rs.{pos.highest_price:.2f})")
                order = self.sell(symbol, current)
                if order:
                    orders.append(order)
            elif pnl_pct >= base_take:
                logger.info(f"TAKE PROFIT triggered for {symbol} ({pnl_pct*100:.1f}%)")
                order = self.sell(symbol, current)
                if order:
                    orders.append(order)
        return orders

    def get_summary(self, prices: dict[str, float]) -> dict:
        """Get portfolio summary."""
        self.refresh_portfolio()
        total = self.portfolio.total_value(prices)
        return {
            "cash": round(self.portfolio.cash, 2),
            "positions_value": round(total - self.portfolio.cash, 2),
            "total_value": round(total, 2),
            "initial_capital": config.INITIAL_CAPITAL,
            "total_return_pct": round(((total - config.INITIAL_CAPITAL) / config.INITIAL_CAPITAL) * 100, 2),
            "realized_pnl": round(self.portfolio.total_realized_pnl, 2),
            "open_positions": len(self.portfolio.positions),
            "total_trades": len(self.portfolio.trade_log),
        }
