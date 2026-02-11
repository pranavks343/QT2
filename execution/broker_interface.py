"""Broker interface inspired by Alpaca-style request/response models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(str, Enum):
    DAY = "day"
    IOC = "ioc"


class OrderStatus(str, Enum):
    NEW = "new"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass
class OrderRequest:
    symbol: str
    qty: int
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: Optional[float] = None
    client_order_id: Optional[str] = None


@dataclass
class Order:
    id: str
    symbol: str
    qty: int
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    submitted_at: str
    filled_at: Optional[str] = None
    filled_avg_price: Optional[float] = None
    client_order_id: Optional[str] = None


@dataclass
class Position:
    symbol: str
    qty: int = 0
    avg_entry_price: float = 0.0
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class BrokerInterface(ABC):
    @abstractmethod
    def submit_order(self, request: OrderRequest, market_price: float) -> Order:
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> Optional[Order]:
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        pass
