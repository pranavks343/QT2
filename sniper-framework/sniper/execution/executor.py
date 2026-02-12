"""Order executor - creates and submits orders."""

from typing import Any

from sniper.execution.cost_model import NSECostModel
from sniper.execution.slippage_model import SlippageModel
from sniper.execution.strategies.market import MarketOrderStrategy


class Executor:
    """
    Executor: creates orders with cost/slippage, supports market/limit.
    """

    def __init__(
        self,
        cost_model: NSECostModel | None = None,
        slippage_model: SlippageModel | None = None,
        default_strategy: str = "market",
    ):
        self.cost_model = cost_model or NSECostModel()
        self.slippage_model = slippage_model or SlippageModel()
        self.market_strategy = MarketOrderStrategy()
        self._default = default_strategy

    def create_order(
        self,
        symbol: str,
        direction: int,
        quantity: int,
        order_type: str = "MARKET",
        price: float | None = None,
        **kwargs: Any,
    ) -> dict:
        """Create an order dict. Price from kwargs for backtest fill."""
        if order_type == "MARKET":
            order = self.market_strategy.create_order(
                symbol=symbol,
                direction=direction,
                quantity=quantity,
                price=price,
                **kwargs,
            )
        else:
            from sniper.execution.strategies.limit import LimitOrderStrategy
            limit = LimitOrderStrategy()
            order = limit.create_order(
                symbol=symbol,
                direction=direction,
                quantity=quantity,
                price=price or 0,
                **kwargs,
            )
        return order
