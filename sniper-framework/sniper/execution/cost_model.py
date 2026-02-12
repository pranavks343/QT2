"""NSE cost model - brokerage, stamp duty, etc."""

from typing import Any


class NSECostModel:
    """
    NSE F&O cost model (simplified).
    """

    def __init__(
        self,
        brokerage_per_lot: float = 20.0,
        stt_pct: float = 0.0625,
        stamp_duty_pct: float = 0.003,
        exchange_txn_pct: float = 0.0002,
    ):
        self.brokerage_per_lot = brokerage_per_lot
        self.stt_pct = stt_pct
        self.stamp_duty_pct = stamp_duty_pct
        self.exchange_txn_pct = exchange_txn_pct

    def compute(self, quantity: int, price: float, is_buy: bool) -> dict:
        """Compute total cost for a trade."""
        notional = quantity * price
        brokerage = self.brokerage_per_lot * (quantity // 25)
        stt = notional * (self.stt_pct / 100)
        stamp = notional * (self.stamp_duty_pct / 100)
        exchange = notional * self.exchange_txn_pct
        total = brokerage + stt + stamp + exchange
        return {
            "brokerage": brokerage,
            "stt": stt,
            "stamp_duty": stamp,
            "exchange": exchange,
            "total": total,
            "notional": notional,
        }
