"""Main orchestrator - wires data, strategy, risk, execution, broker."""

from datetime import datetime

from sniper.core.event_bus import EventBus, Event, EventType
from sniper.core.base_strategy import BaseStrategy
from sniper.data.subscription_manager import SubscriptionManager
from sniper.risk.risk_manager import RiskManager
from sniper.execution.executor import Executor
from sniper.broker.base_broker import BaseBroker


class Algorithm:
    """
    Main orchestrator. Inspired by QuantConnect's Lean Engine.
    Wires together: data -> strategy -> risk -> execution -> broker.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        subscription_manager: SubscriptionManager,
        risk_manager: RiskManager,
        executor: Executor,
        broker: BaseBroker,
        event_bus: EventBus,
    ):
        self.strategy = strategy
        self.subscription_manager = subscription_manager
        self.risk_manager = risk_manager
        self.executor = executor
        self.broker = broker
        self.event_bus = event_bus

        self._setup_event_subscriptions()

    def _setup_event_subscriptions(self) -> None:
        """Wire event flows: signal -> risk -> position size -> order."""
        self.event_bus.subscribe(EventType.SIGNAL_GENERATED, self._handle_signal)
        self.event_bus.subscribe(EventType.RISK_CHECK_PASSED, self._handle_risk_passed)
        self.event_bus.subscribe(EventType.POSITION_SIZED, self._handle_position_sized)
        self.event_bus.subscribe(EventType.ORDER_FILLED, self._handle_order_filled)

    def run_backtest(self, start_date: datetime, end_date: datetime) -> None:
        """Run backtest from start to end."""
        self.strategy.is_backtesting = True
        self.strategy.initialize()

        # Register strategy subscriptions with subscription manager
        for symbol, timeframe in self.strategy.get_subscriptions():
            self.subscription_manager.subscribe(symbol, timeframe)

        self.event_bus.publish(
            Event(
                type=EventType.BACKTEST_STARTED,
                timestamp=start_date,
                data={"start": start_date, "end": end_date},
            )
        )

        self.subscription_manager.start_backtest(start_date, end_date)

        self.strategy.shutdown()
        self.event_bus.publish(
            Event(
                type=EventType.BACKTEST_ENDED,
                timestamp=end_date,
                data={},
            )
        )

    def run_live(self) -> None:
        """Run live trading."""
        self.strategy.is_backtesting = False
        self.strategy.initialize()

        for symbol, timeframe in self.strategy.get_subscriptions():
            self.subscription_manager.subscribe(symbol, timeframe)

        self.event_bus.publish(
            Event(
                type=EventType.LIVE_STARTED,
                timestamp=datetime.now(),
                data={},
            )
        )

        self.subscription_manager.start_live()

    def stop_live(self) -> None:
        """Stop live trading."""
        self.subscription_manager.stop_live()
        self.strategy.shutdown()

        self.event_bus.publish(
            Event(
                type=EventType.LIVE_STOPPED,
                timestamp=datetime.now(),
                data={},
            )
        )

    def _handle_signal(self, event: Event) -> None:
        """Run risk checks on signal."""
        signal = event.data
        if self.risk_manager.can_trade():
            self.event_bus.publish(
                Event(
                    type=EventType.RISK_CHECK_PASSED,
                    timestamp=event.timestamp,
                    data=signal,
                )
            )
        else:
            self.event_bus.publish(
                Event(
                    type=EventType.RISK_CHECK_FAILED,
                    timestamp=event.timestamp,
                    data={
                        "signal": signal,
                        "reason": self.risk_manager.halt_reason or "trading halted",
                    },
                )
            )

    def _handle_risk_passed(self, event: Event) -> None:
        """Compute position size and emit POSITION_SIZED."""
        signal = event.data
        size = self.risk_manager.compute_position_size(signal)

        self.event_bus.publish(
            Event(
                type=EventType.POSITION_SIZED,
                timestamp=event.timestamp,
                data={"signal": signal, "size": size},
            )
        )

    def _handle_position_sized(self, event: Event) -> None:
        """Create order and submit to broker."""
        signal = event.data["signal"]
        size = event.data["size"]

        symbol = signal.get("symbol") or signal.get("metadata", {}).get("symbol", "")
        direction = signal.get("direction", 0)
        quantity = size.get("quantity", size.get("lots", 1) * 25)
        metadata = signal.get("metadata", {})
        price = metadata.get("entry_price")

        order = self.executor.create_order(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            price=price,
        )
        order["metadata"] = metadata
        order["timestamp"] = event.timestamp

        self.broker.place_order(order)

        self.event_bus.publish(
            Event(
                type=EventType.ORDER_SUBMITTED,
                timestamp=event.timestamp,
                data=order,
            )
        )

    def _handle_order_filled(self, event: Event) -> None:
        """Track position when order fills. Position tracker subscribes separately."""
        pass
