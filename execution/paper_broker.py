"""In-memory paper broker with deterministic immediate fills."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional
from uuid import uuid4

from .broker_interface import (
    BrokerInterface,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    Position,
)


class PaperBroker(BrokerInterface):
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}

    def submit_order(self, request: OrderRequest, market_price: float) -> Order:
        order_id = str(uuid4())
        if request.qty <= 0:
            order = Order(
                id=order_id,
                symbol=request.symbol,
                qty=request.qty,
                side=request.side,
                order_type=request.order_type,
                status=OrderStatus.REJECTED,
                submitted_at=datetime.utcnow().isoformat(),
                client_order_id=request.client_order_id,
            )
            self.orders[order_id] = order
            return order

        fill_price = request.limit_price if request.limit_price is not None else market_price
        now = datetime.utcnow().isoformat()
        order = Order(
            id=order_id,
            symbol=request.symbol,
            qty=request.qty,
            side=request.side,
            order_type=request.order_type,
            status=OrderStatus.FILLED,
            submitted_at=now,
            filled_at=now,
            filled_avg_price=float(fill_price),
            client_order_id=request.client_order_id,
        )
        self.orders[order_id] = order
        self._apply_fill(order)
        return order

    def cancel_order(self, order_id: str) -> Optional[Order]:
        order = self.orders.get(order_id)
        if order is None:
            return None
        if order.status == OrderStatus.FILLED:
            return order
        order.status = OrderStatus.CANCELED
        return order

    def get_order(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)

    def get_positions(self) -> Dict[str, Position]:
        return self.positions

    def _apply_fill(self, order: Order) -> None:
        symbol = order.symbol
        pos = self.positions.get(symbol, Position(symbol=symbol))
        signed_qty = order.qty if order.side == OrderSide.BUY else -order.qty

        new_qty = pos.qty + signed_qty
        if new_qty == 0:
            pos.qty = 0
            pos.avg_entry_price = 0.0
        elif pos.qty == 0 or (pos.qty > 0) != (new_qty > 0):
            pos.qty = new_qty
            pos.avg_entry_price = float(order.filled_avg_price or 0.0)
        else:
            total_value = (pos.avg_entry_price * abs(pos.qty)) + ((order.filled_avg_price or 0.0) * order.qty)
            pos.qty = new_qty
            pos.avg_entry_price = total_value / max(abs(pos.qty), 1)

        pos.updated_at = datetime.utcnow().isoformat()
        self.positions[symbol] = pos
