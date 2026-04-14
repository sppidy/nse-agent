"""Paper trading engine — simulates order execution and tracks portfolio."""

import os
import random
import math
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from decimal import Decimal

import config
from logger import logger
from market_calendar import is_market_trading_day
from persistence import read_json, write_json_atomic

# Helper to safely convert to Decimal
def D(value) -> Decimal:
    if value is None:
        return Decimal('0')
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float) and not math.isfinite(value):
        return Decimal('0')
    return Decimal(str(value))

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
    price: Decimal
    timestamp: str
    status: OrderStatus = OrderStatus.FILLED
    order_id: str = ""
    slippage: Decimal = Decimal('0')
    brokerage: Decimal = Decimal('0')

    def __post_init__(self):
        self.price = D(self.price)
        self.slippage = D(self.slippage)
        self.brokerage = D(self.brokerage)

    def fill_price(self) -> Decimal:
        if self.side == OrderSide.BUY:
            return self.price * (Decimal('1') + self.slippage)
        return self.price * (Decimal('1') - self.slippage)

    def total_cost(self) -> Decimal:
        return Decimal(str(self.quantity)) * self.fill_price() + self.brokerage

@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: Decimal
    entry_time: str
    highest_price: Decimal = Decimal('0')
    signal_confidence: Decimal = Decimal('0')
    ai_stop_loss: Decimal | None = None
    ai_target: Decimal | None = None
    dynamic_stop_loss_pct: Decimal = Decimal('0')
    dynamic_take_profit_pct: Decimal = Decimal('0')

    def __post_init__(self):
        self.avg_price = D(self.avg_price)
        self.highest_price = D(self.highest_price)
        self.signal_confidence = D(self.signal_confidence)
        self.ai_stop_loss = D(self.ai_stop_loss) if self.ai_stop_loss is not None else None
        self.ai_target = D(self.ai_target) if self.ai_target is not None else None
        self.dynamic_stop_loss_pct = D(self.dynamic_stop_loss_pct)
        self.dynamic_take_profit_pct = D(self.dynamic_take_profit_pct)

        if self.highest_price == Decimal('0'):
            self.highest_price = self.avg_price
        if self.dynamic_stop_loss_pct <= Decimal('0'):
            self.dynamic_stop_loss_pct = D(config.STOP_LOSS_PCT)
        if self.dynamic_take_profit_pct <= Decimal('0'):
            self.dynamic_take_profit_pct = D(config.TAKE_PROFIT_PCT)

    def current_value(self, current_price: Decimal) -> Decimal:
        return Decimal(str(self.quantity)) * current_price

    def pnl(self, current_price: Decimal) -> Decimal:
        return (current_price - self.avg_price) * Decimal(str(self.quantity))

    def pnl_pct(self, current_price: Decimal) -> Decimal:
        if self.avg_price == Decimal('0'):
            return Decimal('0')
        return ((current_price - self.avg_price) / self.avg_price) * Decimal('100')

@dataclass
class Portfolio:
    cash: Decimal = D(config.INITIAL_CAPITAL)
    positions: dict[str, Position] = field(default_factory=dict)
    orders: list[dict] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)
    total_realized_pnl: Decimal = Decimal('0')

    def __post_init__(self):
        self.cash = D(self.cash)
        self.total_realized_pnl = D(self.total_realized_pnl)

        normalized_positions: dict[str, Position] = {}
        for sym, pos in self.positions.items():
            if isinstance(pos, Position):
                normalized_positions[sym] = pos
            elif isinstance(pos, dict):
                normalized_positions[sym] = Position(**pos)
        self.positions = normalized_positions

        if self.orders is None:
            self.orders = []
        if self.trade_log is None:
            self.trade_log = []

    @staticmethod
    def _safe_price(value: float | Decimal | None, fallback: Decimal) -> Decimal:
        if value is None:
            return fallback
        try:
            price = float(value)
        except (TypeError, ValueError):
            return fallback
        if not math.isfinite(price) or price <= 0:
            return fallback
        return D(value)

    def total_value(self, prices: dict[str, float]) -> Decimal:
        positions_value = sum(
            pos.current_value(self._safe_price(prices.get(sym), pos.avg_price))
            for sym, pos in self.positions.items()
        )
        if positions_value == 0:  # Handle empty sum returning int 0
            positions_value = Decimal('0')
        return self.cash + positions_value

    @staticmethod
    def _resolve_path(filepath: str) -> str:
        if os.path.isabs(filepath):
            return filepath
        return os.path.join(config.PROJECT_DIR, filepath)

    def save(self, filepath: str = "portfolio.json"):
        filepath = self._resolve_path(filepath)
        
        def _decimal_to_float(obj):
            if isinstance(obj, Decimal):
                return float(obj)
            if isinstance(obj, dict):
                return {k: _decimal_to_float(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_decimal_to_float(v) for v in obj]
            return obj

        pos_dict = {}
        for sym, pos in self.positions.items():
            pos_dict[sym] = _decimal_to_float(asdict(pos))

        data = {
            "cash": float(self.cash),
            "positions": pos_dict,
            "orders": _decimal_to_float(self.orders),
            "trade_log": _decimal_to_float(self.trade_log),
            "total_realized_pnl": float(self.total_realized_pnl),
        }
        write_json_atomic(filepath, data)

    @classmethod
    def load(cls, filepath: str = "portfolio.json") -> "Portfolio":
        filepath = cls._resolve_path(filepath)
        data = read_json(filepath, default=None)
        if not isinstance(data, dict):
            return cls()
        portfolio = cls(
            cash=D(data.get("cash", config.INITIAL_CAPITAL)),
            total_realized_pnl=D(data.get("total_realized_pnl", 0)),
        )
        for sym, pos_data in data.get("positions", {}).items():
            clean_data = {}
            for k, v in pos_data.items():
                if isinstance(v, float) or isinstance(v, int):
                    if k != 'quantity':
                        clean_data[k] = D(v)
                    else:
                        clean_data[k] = int(v)
                else:
                    clean_data[k] = v
            portfolio.positions[sym] = Position(**clean_data)
            
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
    def _clamp(value: Decimal, low: Decimal, high: Decimal) -> Decimal:
        return max(low, min(value, high))

    def refresh_portfolio(self) -> None:
        if not self._disk_sync_enabled:
            return
        latest = Portfolio.load(self.filepath)
        self.portfolio = latest
        self._order_counter = max(self._order_counter, len(self.portfolio.orders))

    def _capital_utilization_floor_pct(self, max_position_size_pct: Decimal) -> Decimal:
        initial_capital = max(D(config.INITIAL_CAPITAL), Decimal('1'))
        cash_ratio = self.portfolio.cash / initial_capital
        deploy_gap = max(Decimal('0'), cash_ratio - (Decimal('1') - D(config.CAPITAL_DEPLOYMENT_TARGET_PCT)))
        slot_budget = deploy_gap / max(Decimal(str(config.MAX_OPEN_POSITIONS)), Decimal('1'))
        floor_pct = max(D(config.CAPITAL_UTILIZATION_MIN_BET_PCT), slot_budget)
        return self._clamp(floor_pct, Decimal('0'), max_position_size_pct)

    def _dynamic_risk_levels(
        self,
        entry_price: Decimal,
        confidence: Decimal,
        ai_signal: dict | None,
    ) -> tuple[Decimal, Decimal, Decimal | None, Decimal | None]:
        safe_conf = self._clamp(confidence, Decimal('0'), Decimal('1'))
        stop_pct = D(config.STOP_LOSS_PCT)
        take_pct = D(config.TAKE_PROFIT_PCT)

        ai_stop = None
        ai_target = None
        if ai_signal:
            ai_stop_raw = ai_signal.get("stop_loss")
            ai_target_raw = ai_signal.get("target")
            ai_stop_val = D(ai_stop_raw) if ai_stop_raw is not None else None
            ai_target_val = D(ai_target_raw) if ai_target_raw is not None else None

            if ai_stop_val is not None and ai_stop_val > Decimal('0') and ai_stop_val < entry_price:
                ai_stop = ai_stop_val
                stop_pct = (stop_pct + ((entry_price - ai_stop_val) / entry_price)) / Decimal('2')
            if ai_target_val is not None and ai_target_val > entry_price:
                ai_target = ai_target_val
                take_pct = (take_pct + ((ai_target_val - entry_price) / entry_price)) / Decimal('2')

        conf_shift = (Decimal('0.5') - safe_conf) * D(config.TRAILING_CONFIDENCE_SCALE)
        stop_pct *= (Decimal('1') + conf_shift)
        take_pct *= (Decimal('1') - conf_shift)

        stop_pct = self._clamp(stop_pct, D(config.MIN_STOP_LOSS_PCT), D(config.MAX_STOP_LOSS_PCT))
        take_pct = self._clamp(take_pct, D(config.MIN_TAKE_PROFIT_PCT), D(config.MAX_TAKE_PROFIT_PCT))
        return stop_pct, take_pct, ai_stop, ai_target

    def buy(
        self,
        symbol: str,
        price: float | Decimal,
        quantity: int | None = None,
        confidence: float | Decimal = 0.0,
        max_position_size_pct: float | Decimal | None = None,
        ai_signal: dict | None = None,
    ) -> Order | None:
        """Place a simulated buy order."""
        self.refresh_portfolio()
        if not is_market_trading_day():
            logger.warning(f"Cannot buy {symbol}: non-trading day (weekend/holiday)")
            return None

        price = D(price)
        confidence = D(confidence)
        if max_position_size_pct is None:
            max_position_size_pct = D(config.MAX_POSITION_SIZE_PCT)
        else:
            max_position_size_pct = D(max_position_size_pct)

        # Simulate dynamic slippage (liquidity constraints)
        slippage = D(config.SLIPPAGE_PCT) * D(random.uniform(0.5, 2.0))
        fill_price = price * (Decimal('1') + slippage)
        brokerage = D(config.BROKERAGE_PER_ORDER)

        # Auto-calculate quantity if not specified
        if quantity is None:
            # Phase 2: Kelly Criterion Dynamic Position Sizing
            if confidence > Decimal('0'):
                W = confidence
                # Reward/Risk Ratio (e.g., 3% / 2% = 1.5)
                R = D(config.TAKE_PROFIT_PCT) / D(config.STOP_LOSS_PCT)
                # Kelly fraction: f = W - ((1 - W) / R)
                kelly_fraction = W - ((Decimal('1') - W) / R)
                # Half-Kelly for safety
                half_kelly = kelly_fraction / Decimal('2')
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
                f"(need Rs.{float(fill_price):.2f}, have Rs.{float(self.portfolio.cash):.2f})"
            )
            return None

        total_cost = Decimal(str(quantity)) * fill_price + brokerage
        if total_cost > self.portfolio.cash:
            # Reduce quantity to fit
            quantity = int((self.portfolio.cash - brokerage) / fill_price)
            if quantity <= 0:
                logger.warning(f"Cannot buy {symbol}: insufficient funds")
                return None
            total_cost = Decimal(str(quantity)) * fill_price + brokerage

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
            pos.avg_price = (pos.avg_price * Decimal(str(pos.quantity)) + fill_price * Decimal(str(quantity))) / Decimal(str(total_qty))
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

        # Convert order fields to float before dicting for orders list
        order_dict = asdict(order)
        order_dict['price'] = float(order_dict['price'])
        order_dict['slippage'] = float(order_dict['slippage'])
        order_dict['brokerage'] = float(order_dict['brokerage'])
        self.portfolio.orders.append(order_dict)
        if self._disk_sync_enabled:
            self.portfolio.save(self.filepath)
        logger.info(f"BUY {quantity}x {symbol} @ Rs.{float(fill_price):.2f} = Rs.{float(total_cost):.2f}")
        return order

    def sell(self, symbol: str, price: float | Decimal, quantity: int | None = None) -> Order | None:
        """Place a simulated sell order."""
        self.refresh_portfolio()
        if not is_market_trading_day():
            logger.warning(f"Cannot sell {symbol}: non-trading day (weekend/holiday)")
            return None

        price = D(price)

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
        slippage = D(config.SLIPPAGE_PCT) * D(random.uniform(0.5, 2.0))
        fill_price = price * (Decimal('1') - slippage)
        brokerage = D(config.BROKERAGE_PER_ORDER)

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

        proceeds = Decimal(str(quantity)) * fill_price - brokerage
        self.portfolio.cash += proceeds

        # Calculate realized P&L
        pnl = (fill_price - pos.avg_price) * Decimal(str(quantity))
        self.portfolio.total_realized_pnl += pnl

        self.portfolio.trade_log.append({
            "symbol": symbol,
            "side": "SELL",
            "quantity": quantity,
            "entry_price": float(pos.avg_price),
            "exit_price": float(fill_price),
            "pnl": round(float(pnl), 2),
            "pnl_pct": round(float(((fill_price - pos.avg_price) / pos.avg_price) * Decimal('100')), 2),
            "timestamp": order.timestamp,
        })

        if quantity >= pos.quantity:
            del self.portfolio.positions[symbol]
        else:
            pos.quantity -= quantity

        order_dict = asdict(order)
        order_dict['price'] = float(order_dict['price'])
        order_dict['slippage'] = float(order_dict['slippage'])
        order_dict['brokerage'] = float(order_dict['brokerage'])
        self.portfolio.orders.append(order_dict)
        
        if self._disk_sync_enabled:
            self.portfolio.save(self.filepath)
        pnl_str = f"+Rs.{float(pnl):.2f}" if pnl >= Decimal('0') else f"-Rs.{abs(float(pnl)):.2f}"
        logger.info(f"SELL {quantity}x {symbol} @ Rs.{float(fill_price):.2f} | P&L: {pnl_str}")
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
            
            current_d = D(current)
                
            # Update highest price seen for trailing stop
            pos.highest_price = max(pos.highest_price, current_d)
            if self._disk_sync_enabled:
                self.portfolio.save(self.filepath)

            pnl_pct = pos.pnl_pct(current_d) / Decimal('100')

            base_stop = pos.dynamic_stop_loss_pct if config.DYNAMIC_TRAILING_ENABLED else D(config.STOP_LOSS_PCT)
            base_take = pos.dynamic_take_profit_pct if config.DYNAMIC_TRAILING_ENABLED else D(config.TAKE_PROFIT_PCT)
            runup_pct = max(Decimal('0'), (pos.highest_price - pos.avg_price) / pos.avg_price) if pos.avg_price > Decimal('0') else Decimal('0')
            tightened_stop = max(D(config.MIN_STOP_LOSS_PCT), base_stop - (runup_pct * D(config.TRAILING_PROFIT_LOCK_SCALE)))
            trailing_loss_pct = (current_d - pos.highest_price) / pos.highest_price

            if trailing_loss_pct <= -tightened_stop:
                logger.info(f"TRAILING STOP LOSS triggered for {symbol} ({float(trailing_loss_pct)*100:.1f}% from peak of Rs.{float(pos.highest_price):.2f})")
                order = self.sell(symbol, current_d)
                if order:
                    orders.append(order)
            elif pnl_pct >= base_take:
                logger.info(f"TAKE PROFIT triggered for {symbol} ({float(pnl_pct)*100:.1f}%)")
                order = self.sell(symbol, current_d)
                if order:
                    orders.append(order)
        return orders

    def get_summary(self, prices: dict[str, float]) -> dict:
        """Get portfolio summary."""
        self.refresh_portfolio()
        total = self.portfolio.total_value(prices)
        return {
            "cash": round(float(self.portfolio.cash), 2),
            "positions_value": round(float(total - self.portfolio.cash), 2),
            "total_value": round(float(total), 2),
            "initial_capital": config.INITIAL_CAPITAL,
            "total_return_pct": round(float(((total - D(config.INITIAL_CAPITAL)) / D(config.INITIAL_CAPITAL)) * Decimal('100')), 2),
            "realized_pnl": round(float(self.portfolio.total_realized_pnl), 2),
            "open_positions": len(self.portfolio.positions),
            "total_trades": len(self.portfolio.trade_log),
        }
