"""
SNIPER FRAMEWORK - EXECUTION COST MODEL
Models all real NSE derivatives trading costs:
  - STT (Securities Transaction Tax) — the big one for options
  - Brokerage (flat per order)
  - Exchange transaction charges
  - SEBI charges
  - Stamp duty
  - GST on brokerage
  - Slippage (bid-ask spread + market impact)

Also: STT trap warning for deep ITM options bought and exercised.
"""

from dataclasses import dataclass
from config import COSTS


@dataclass
class TradeTicket:
    """All the info needed to compute cost for one trade."""
    direction: int          # +1 long, -1 short
    entry_price: float      # Option premium at entry (per unit)
    exit_price: float       # Option premium at exit (per unit)
    lots: int               # Number of lots
    lot_size: int           # Units per lot (75 for Nifty, 30 for BankNifty)
    is_option: bool = True  # True for options, False for futures
    exercised: bool = False # True if held to expiry and exercised


@dataclass
class CostBreakdown:
    """Full cost breakdown for a trade (in INR)."""
    entry_stt: float
    exit_stt: float
    entry_brokerage: float
    exit_brokerage: float
    entry_exchange: float
    exit_exchange: float
    entry_sebi: float
    exit_sebi: float
    entry_stamp_duty: float
    exit_stamp_duty: float
    gst: float
    entry_slippage: float
    exit_slippage: float
    total_cost: float
    cost_per_lot: float
    cost_as_pct_of_premium: float
    stt_trap_warning: bool


class ExecutionCostModel:
    """
    Compute realistic all-in execution costs for NSE F&O trades.
    Reference: NSE/SEBI charge schedule (as of 2024)
    """

    def compute(self, ticket: TradeTicket) -> CostBreakdown:
        qty = ticket.lots * ticket.lot_size
        entry_value = ticket.entry_price * qty
        exit_value = ticket.exit_price * qty

        # ── STT ──────────────────────────────────────────────────────────────
        # Options: 0.05% only on SELL (premium value) under normal exit
        # If exercised: 0.125% on INTRINSIC value — the STT trap
        if ticket.is_option:
            entry_stt = 0.0  # Buy side: no STT on options
            if ticket.exercised:
                # STT trap: charged on full intrinsic value, not premium
                # This can wipe the entire profit on cheap options
                exit_stt = exit_value * 0.00125  # 0.125% on exercise
                stt_trap = True
            else:
                exit_stt = exit_value * COSTS["stt_sell_pct"]
                stt_trap = False
        else:
            # Futures: 0.01% on both sides
            entry_stt = entry_value * 0.0001
            exit_stt = exit_value * 0.0001
            stt_trap = False

        # ── BROKERAGE ─────────────────────────────────────────────────────────
        # Flat ₹20 per order (Zerodha/Upstox model) with 0.03% cap check
        entry_brokerage = min(COSTS["brokerage_per_order"], entry_value * 0.0003)
        exit_brokerage = min(COSTS["brokerage_per_order"], exit_value * 0.0003)

        # ── EXCHANGE TRANSACTION CHARGES ──────────────────────────────────────
        entry_exchange = entry_value * COSTS["exchange_txn_pct"]
        exit_exchange = exit_value * COSTS["exchange_txn_pct"]

        # ── SEBI CHARGES ──────────────────────────────────────────────────────
        entry_sebi = entry_value * COSTS["sebi_charges_pct"]
        exit_sebi = exit_value * COSTS["sebi_charges_pct"]

        # ── STAMP DUTY (buy side only) ────────────────────────────────────────
        entry_stamp = entry_value * COSTS["stamp_duty_buy_pct"]
        exit_stamp = 0.0  # Stamp duty only on buy

        # ── GST (18% on brokerage + exchange charges) ─────────────────────────
        taxable = (entry_brokerage + exit_brokerage + entry_exchange + exit_exchange)
        gst = taxable * COSTS["gst_on_brokerage_pct"]

        # ── SLIPPAGE ──────────────────────────────────────────────────────────
        slippage_rate = COSTS["slippage_bps"] / 10_000
        entry_slippage = entry_value * slippage_rate
        exit_slippage = exit_value * slippage_rate

        # ── TOTAL ─────────────────────────────────────────────────────────────
        total = (entry_stt + exit_stt + entry_brokerage + exit_brokerage +
                 entry_exchange + exit_exchange + entry_sebi + exit_sebi +
                 entry_stamp + exit_stamp + gst + entry_slippage + exit_slippage)

        avg_premium = (ticket.entry_price + ticket.exit_price) / 2
        premium_total = avg_premium * qty

        return CostBreakdown(
            entry_stt=round(entry_stt, 2),
            exit_stt=round(exit_stt, 2),
            entry_brokerage=round(entry_brokerage, 2),
            exit_brokerage=round(exit_brokerage, 2),
            entry_exchange=round(entry_exchange, 2),
            exit_exchange=round(exit_exchange, 2),
            entry_sebi=round(entry_sebi, 2),
            exit_sebi=round(exit_sebi, 2),
            entry_stamp_duty=round(entry_stamp, 2),
            exit_stamp_duty=round(exit_stamp, 2),
            gst=round(gst, 2),
            entry_slippage=round(entry_slippage, 2),
            exit_slippage=round(exit_slippage, 2),
            total_cost=round(total, 2),
            cost_per_lot=round(total / max(ticket.lots, 1), 2),
            cost_as_pct_of_premium=round(total / max(premium_total, 1) * 100, 4),
            stt_trap_warning=stt_trap,
        )

    def net_pnl(self, ticket: TradeTicket) -> dict:
        """
        Compute gross P&L and net P&L after all costs.
        Returns full breakdown dict.
        """
        qty = ticket.lots * ticket.lot_size
        if ticket.direction == 1:
            gross_pnl = (ticket.exit_price - ticket.entry_price) * qty
        else:
            gross_pnl = (ticket.entry_price - ticket.exit_price) * qty

        costs = self.compute(ticket)
        net = gross_pnl - costs.total_cost

        return {
            "gross_pnl": round(gross_pnl, 2),
            "total_cost": costs.total_cost,
            "net_pnl": round(net, 2),
            "cost_breakdown": costs,
            "is_profitable": net > 0,
            "break_even_move_pct": round(costs.total_cost / (ticket.entry_price * qty) * 100, 4),
        }

    def is_tradeable(self, premium: float) -> bool:
        """
        Reject trades where premium is too low — STT cost alone
        can exceed potential profit (the 'STT trap').
        """
        return premium >= COSTS["min_trade_premium"]

    def cost_summary_str(self, costs: CostBreakdown) -> str:
        lines = [
            f"  STT (entry/exit):     ₹{costs.entry_stt:,.2f} / ₹{costs.exit_stt:,.2f}",
            f"  Brokerage:            ₹{costs.entry_brokerage + costs.exit_brokerage:,.2f}",
            f"  Exchange charges:     ₹{costs.entry_exchange + costs.exit_exchange:,.2f}",
            f"  Stamp duty:           ₹{costs.entry_stamp_duty:,.2f}",
            f"  GST:                  ₹{costs.gst:,.2f}",
            f"  Slippage:             ₹{costs.entry_slippage + costs.exit_slippage:,.2f}",
            f"  {'─'*40}",
            f"  TOTAL COST:           ₹{costs.total_cost:,.2f}",
            f"  Cost as % of premium: {costs.cost_as_pct_of_premium:.2f}%",
        ]
        if costs.stt_trap_warning:
            lines.append("  ⚠️  STT TRAP WARNING: Option exercised at expiry!")
        return "\n".join(lines)
