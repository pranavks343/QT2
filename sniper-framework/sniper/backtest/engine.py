"""Backtest engine - orchestrates strategy run and collects results."""

from datetime import datetime
from typing import Type, Any

from sniper.core.event_bus import EventBus, Event, EventType
from sniper.core.base_strategy import BaseStrategy
from sniper.core.algorithm import Algorithm
from sniper.data.base_provider import BaseDataProvider
from sniper.data.subscription_manager import SubscriptionManager
from sniper.data.synthetic_provider import SyntheticProvider
from sniper.risk.risk_manager import RiskManager
from sniper.execution.executor import Executor
from sniper.broker.paper_broker import PaperBroker

from sniper.backtest.results import BacktestResults


class BacktestEngine:
    """
    Backtest engine: wires components, runs backtest, collects fills and P&L.
    """

    def __init__(
        self,
        strategy: BaseStrategy | Type[BaseStrategy],
        data_provider: BaseDataProvider | None = None,
        initial_capital: float = 500000.0,
        event_bus: EventBus | None = None,
    ):
        self.strategy_class = strategy if isinstance(strategy, type) else type(strategy)
        self.data_provider = data_provider or SyntheticProvider(seed=42)
        self.initial_capital = initial_capital
        self.event_bus = event_bus or EventBus()

        self._strategy: BaseStrategy | None = None
        self._algorithm: Algorithm | None = None
        self._broker: PaperBroker | None = None
        self._risk_manager: RiskManager | None = None
        self._positions: dict[str, dict] = {}
        self._trades: list[dict] = []
        self._equity_curve: list[tuple[datetime, float]] = []

    def run(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> BacktestResults:
        """Run backtest and return results."""
        self._positions = {}
        self._trades = []
        self._equity_curve = [(start_date, self.initial_capital)]

        self.event_bus.subscribe(EventType.ORDER_FILLED, self._on_order_filled)

        strategy = self.strategy_class(self.event_bus)
        self._strategy = strategy
        sub_manager = SubscriptionManager(self.event_bus, self.data_provider)
        risk_manager = RiskManager(self.event_bus, self.initial_capital)
        self._risk_manager = risk_manager
        executor = Executor()
        broker = PaperBroker(event_bus=self.event_bus)
        self._broker = broker

        algorithm = Algorithm(
            strategy=strategy,
            subscription_manager=sub_manager,
            risk_manager=risk_manager,
            executor=executor,
            broker=broker,
            event_bus=self.event_bus,
        )
        self._algorithm = algorithm

        algorithm.run_backtest(start_date, end_date)

        self.event_bus.unsubscribe(EventType.ORDER_FILLED, self._on_order_filled)

        return BacktestResults(
            trades=self._trades,
            equity_curve=self._equity_curve,
            initial_capital=self.initial_capital,
            fills=self._broker.fills if self._broker else [],
        )

    def _on_order_filled(self, event: Event) -> None:
        """Track positions and compute P&L from fills."""
        fill = event.data
        symbol = fill.get("symbol", "")
        direction = fill.get("direction", 0)
        quantity = fill.get("quantity", 0)
        price = fill.get("fill_price", 0)
        ts = fill.get("timestamp", event.timestamp)

        if not symbol or quantity <= 0:
            return

        pos = self._positions.get(symbol, {"quantity": 0, "avg_price": 0, "direction": 0})

        if pos["quantity"] == 0:
            pos = {"quantity": quantity, "avg_price": price, "direction": direction}
            self._positions[symbol] = pos
        elif (pos["direction"] > 0 and direction > 0) or (pos["direction"] < 0 and direction < 0):
            total_qty = pos["quantity"] + quantity
            pos["avg_price"] = (
                (pos["avg_price"] * pos["quantity"] + price * quantity) / total_qty
                if total_qty else pos["avg_price"]
            )
            pos["quantity"] = total_qty
        else:
            close_qty = min(pos["quantity"], quantity)
            entry = pos["avg_price"]
            pnl = (price - entry) * close_qty * pos["direction"]
            self._trades.append({
                "symbol": symbol,
                "entry_price": entry,
                "exit_price": price,
                "quantity": close_qty,
                "direction": pos["direction"],
                "pnl": pnl,
                "entry_time": self._equity_curve[-1][0] if self._equity_curve else ts,
                "exit_time": ts,
            })

            if self._risk_manager:
                self._risk_manager.capital += pnl
                self._risk_manager.update_capital(self._risk_manager.capital)
            self._equity_curve.append((ts, self._equity_curve[-1][1] + pnl))

            pos["quantity"] -= close_qty
            if pos["quantity"] <= 0:
                del self._positions[symbol]
            else:
                self._positions[symbol] = pos

            if quantity > close_qty:
                remaining = quantity - close_qty
                self._positions[symbol] = {
                    "quantity": remaining,
                    "avg_price": price,
                    "direction": direction,
                }
