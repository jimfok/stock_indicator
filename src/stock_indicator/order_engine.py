"""
Phase 2: Order Engine — bar processing, fill evaluation, position lifecycle.

Architecture doc: docs/findings/order_model_architecture.md

Design decisions (confirmed):
  - Gap fills (open beyond TP/SL)   → fill at open price
  - TP vs exit_signal same bar       → TP wins
  - TP/SL modification mid-trade    → no (fixed after placement)
  - File structure                   → new files

Dependencies
------------
order_model.py — Signal, Order, Position, Trade, OrderType, OrderStatus, ExitReason
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Tuple

import pandas

from .order_model import (
    ExitReason,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Signal,
    SignalType,
    Trade,
)


# ─────────────────────────────────────────────────────────────────────────────
# Commission (copied from simulator.py for self-contained module)
# ─────────────────────────────────────────────────────────────────────────────

BROKER_PLATFORM_FEE = 0.0
SETTLEMENT_FEE_RATE = 0.00013
SEC_FEE_RATE        = 0.000013
TAF_FEE_PER_SHARE   = 0.000195
MAX_TAF_FEE         = 9.79


def round_lot_commission(shares: int, price: float) -> float:
    """Round-lot only commission model (same as current simulator)."""
    if shares <= 0 or price <= 0:
        return 0.0
    proceeds = shares * price
    broker_platform = BROKER_PLATFORM_FEE
    settlement = proceeds * SETTLEMENT_FEE_RATE
    sec_fee   = proceeds * SEC_FEE_RATE
    taf_fee   = min(MAX_TAF_FEE, max(0.01, shares * TAF_FEE_PER_SHARE))
    return broker_platform + settlement + sec_fee + taf_fee


# ─────────────────────────────────────────────────────────────────────────────
# Order evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_bar(
    bar: pandas.Series,
    position: Optional[Position],
    open_orders: List[Order],
    pending_signal_exit: bool,
    min_hold_bars: int,
) -> Tuple[List[Order], List[Trade], Optional[Position], bool]:
    """
    Evaluate one bar for a single symbol.

    Parameters
    ----------
    bar : pandas.Series
        Current bar (index = date, columns = OHLCV).
    position : Position | None
        Current open position for this symbol.
    open_orders : List[Order]
        All pending orders for this symbol.
    pending_signal_exit : bool
        True if an exit signal was received on the previous bar
        and a SELL market order is pending (to be filled at today's open).
    min_hold_bars : int
        Minimum bars to hold before exit signal can trigger a SELL.

    Returns
    -------
    Tuple[List[Order], List[Trade], Optional[Position], bool]
        - filled_orders : orders that were filled this bar
        - completed_trades : Trade objects created this bar
        - position : updated position (or None if closed)
        - pending_signal_exit : carries forward if still not filled
    """
    filled_orders:      List[Order]  = []
    completed_trades:  List[Trade]  = []
    current_position   = position
    signal_exit_fired  = pending_signal_exit

    if current_position is None:
        return filled_orders, completed_trades, current_position, signal_exit_fired

    # ── 1. Check SELL order first (signal exit) ──────────────────────────
    # If pending from previous bar, fill at today's open.
    # Priority: TP > SL > SIGNAL_EXIT.
    if signal_exit_fired:
        # Check SL first (in case both are set — should not happen, but safe)
        sl_order = _find_pending_sl(open_orders)
        if sl_order is not None:
            sl_triggered, trade = _try_fill_sl(
                sl_order, bar, current_position
            )
            if sl_triggered:
                filled_orders.append(sl_order)
                completed_trades.append(trade)
                return filled_orders, completed_trades, None, False

    # ── 2. Check TP order ─────────────────────────────────────────────────
    tp_order = _find_pending_tp(open_orders)
    if tp_order is not None:
        tp_triggered, trade = _try_fill_tp(tp_order, bar, current_position)
        if tp_triggered:
            filled_orders.append(tp_order)
            completed_trades.append(trade)
            # SELL order is now moot (position closed by TP)
            return filled_orders, completed_trades, None, False

    # ── 3. Check SL order ─────────────────────────────────────────────────
    sl_order = _find_pending_sl(open_orders)
    if sl_order is not None:
        sl_triggered, trade = _try_fill_sl(sl_order, bar, current_position)
        if sl_triggered:
            filled_orders.append(sl_order)
            completed_trades.append(trade)
            return filled_orders, completed_trades, None, False

    # ── 4. Check SELL market order (signal exit) ────────────────────────────
    # Requires min_hold_bars to have passed since entry.
    # Filled at bar open (gap-fill rule applies: if bar opened below entry,
    # still fill at open — but since this is a SELL market order, the fill
    # is always at open regardless).
    if signal_exit_fired:
        bars_held = _bars_between(current_position.entry_date, bar.name)
        if bars_held >= min_hold_bars:
            sell_trade = _close_via_signal_exit(bar, current_position)
            filled_orders.append(_make_filled_sell_order(bar, current_position))
            completed_trades.append(sell_trade)
            return filled_orders, completed_trades, None, False

    # ── 5. No fill this bar — update position tracking ────────────────────
    _update_position_tracking(bar, current_position)
    return filled_orders, completed_trades, current_position, signal_exit_fired


# ─────────────────────────────────────────────────────────────────────────────
# Order fill helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_pending_tp(orders: List[Order]) -> Optional[Order]:
    for o in orders:
        if o.order_type == OrderType.TAKE_PROFIT and o.is_active():
            return o
    return None


def _find_pending_sl(orders: List[Order]) -> Optional[Order]:
    for o in orders:
        if o.order_type == OrderType.STOP_LOSS and o.is_active():
            return o
    return None


def _try_fill_tp(
    tp_order: Order,
    bar: pandas.Series,
    position: Position,
) -> Tuple[bool, Trade]:
    """
    Try to fill a TP order this bar.

    Logic:
    - TP triggers if bar high >= trigger_price.
    - If bar opens above trigger_price (gap up), fill at bar open
      (gap-fill rule).
    - Otherwise fill at trigger_price.
    """
    if tp_order.trigger_price is None:
        return False, _empty_trade(position)

    trigger = tp_order.trigger_price
    open_p  = float(bar["open"])
    high_p  = float(bar["high"])

    # Gap-fill: bar opened above trigger
    if open_p >= trigger:
        tp_order.status    = OrderStatus.FILLED
        tp_order.fill_date = bar.name
        tp_order.fill_price = open_p
        return True, _close_via_tp(bar, position, open_p)

    # Normal: bar high reached trigger
    if high_p >= trigger:
        tp_order.status    = OrderStatus.FILLED
        tp_order.fill_date = bar.name
        tp_order.fill_price = trigger  # Fill at trigger, not high
        return True, _close_via_tp(bar, position, trigger)

    return False, _empty_trade(position)


def _try_fill_sl(
    sl_order: Order,
    bar: pandas.Series,
    position: Position,
) -> Tuple[bool, Trade]:
    """
    Try to fill an SL order this bar.

    Logic:
    - SL only active after sl_activation_date.
    - SL triggers if bar low <= trigger_price.
    - If bar opens below trigger_price (gap down), fill at bar open
      (gap-fill rule).
    - Otherwise fill at trigger_price.
    """
    if sl_order.trigger_price is None:
        return False, _empty_trade(position)

    # SL not yet active
    if sl_order.sl_activation is not None and bar.name < sl_order.sl_activation:
        return False, _empty_trade(position)

    trigger = sl_order.trigger_price
    open_p  = float(bar["open"])
    low_p   = float(bar["low"])

    # Gap-fill: bar opened below stop
    if open_p <= trigger:
        sl_order.status    = OrderStatus.FILLED
        sl_order.fill_date = bar.name
        sl_order.fill_price = open_p
        return True, _close_via_sl(bar, position, open_p)

    # Normal: bar low reached stop
    if low_p <= trigger:
        sl_order.status    = OrderStatus.FILLED
        sl_order.fill_date = bar.name
        sl_order.fill_price = trigger  # Fill at stop, not low
        return True, _close_via_sl(bar, position, trigger)

    return False, _empty_trade(position)


def _close_via_tp(bar: pandas.Series, position: Position, fill_price: float) -> Trade:
    holding = _bars_between(position.entry_date, bar.name)
    pnl     = (fill_price - position.entry_price) * position.quantity
    trade   = Trade(
        symbol=position.symbol,
        entry_date=position.entry_date,
        entry_price=position.entry_price,
        exit_date=bar.name,
        exit_price=fill_price,
        quantity=position.quantity,
        pnl=pnl,
        holding_period=holding,
        exit_reason=ExitReason.TAKE_PROFIT,
        tp_triggered=True,
        tp_trigger_price=position.tp_trigger,
        sl_trigger_price=position.sl_trigger,
        set_label=getattr(position, "set_label", None),
    )
    return trade


def _close_via_sl(bar: pandas.Series, position: Position, fill_price: float) -> Trade:
    holding = _bars_between(position.entry_date, bar.name)
    pnl     = (fill_price - position.entry_price) * position.quantity
    trade   = Trade(
        symbol=position.symbol,
        entry_date=position.entry_date,
        entry_price=position.entry_price,
        exit_date=bar.name,
        exit_price=fill_price,
        quantity=position.quantity,
        pnl=pnl,
        holding_period=holding,
        exit_reason=ExitReason.STOP_LOSS,
        sl_triggered=True,
        tp_trigger_price=position.tp_trigger,
        sl_trigger_price=position.sl_trigger,
        set_label=getattr(position, "set_label", None),
    )
    return trade


def _close_via_signal_exit(bar: pandas.Series, position: Position) -> Trade:
    fill_price = float(bar["open"])  # SELL market order fills at open
    holding    = _bars_between(position.entry_date, bar.name)
    pnl        = (fill_price - position.entry_price) * position.quantity
    trade      = Trade(
        symbol=position.symbol,
        entry_date=position.entry_date,
        entry_price=position.entry_price,
        exit_date=bar.name,
        exit_price=fill_price,
        quantity=position.quantity,
        pnl=pnl,
        holding_period=holding,
        exit_reason=ExitReason.SIGNAL_EXIT,
        signal_exit_fired=True,
        tp_trigger_price=position.tp_trigger,
        sl_trigger_price=position.sl_trigger,
        set_label=getattr(position, "set_label", None),
    )
    return trade


def _make_filled_sell_order(bar: pandas.Series, position: Position) -> Order:
    return Order(
        order_id=uuid.uuid4().hex[:12],
        order_type=OrderType.SELL,
        symbol=position.symbol,
        date_placed=bar.name,
        expected_fill=bar.name,
        quantity=position.quantity,
        fill_date=bar.name,
        fill_price=float(bar["open"]),
        status=OrderStatus.FILLED,
    )


def _update_position_tracking(bar: pandas.Series, position: Position) -> None:
    """Update unrealized P&L and bar excursion on each bar."""
    current_price = float(bar["close"])
    high_p        = float(bar["high"])
    low_p         = float(bar["low"])

    position.update_unrealized(current_price)
    position.update_high(high_p)

    # Bar excursion (high/low relative to entry)
    high_pct = (high_p - position.entry_price) / position.entry_price
    low_pct  = (low_p  - position.entry_price) / position.entry_price

    if position.bar_excursions is None:
        position.bar_excursions = []  # type: ignore[attr-defined]
    position.bar_excursions.append((bar.name, high_pct, low_pct))  # type: ignore[attr-defined]

    # Track MFE/MAE
    current_holding = len(position.bar_excursions or [])
    if position.max_favorable_pct is None or high_pct > position.max_favorable_pct:  # type: ignore[attr-defined]
        position.max_favorable_pct = high_pct  # type: ignore[attr-defined]
        position.mfe_date = bar.name  # type: ignore[attr-defined]
    if position.max_adverse_pct is None or low_pct < position.max_adverse_pct:  # type: ignore[attr-defined]
        position.max_adverse_pct = low_pct  # type: ignore[attr-defined]
        position.mae_date = bar.name  # type: ignore[attr-defined]


def _bars_between(start: pandas.Timestamp, end: pandas.Timestamp) -> int:
    """Business-day count between two dates (consistent with upstream convention).
    Returns 0 for negative or same-day ranges."""
    if end <= start:
        return 0
    return max(0, len(pandas.bdate_range(start, end)) - 1)


def _empty_trade(position: Position) -> Trade:
    """Placeholder when no fill occurred."""
    return Trade(
        symbol=position.symbol,
        entry_date=position.entry_date,
        entry_price=position.entry_price,
        quantity=position.quantity,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Order creation helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_buy_order(
    symbol: str,
    signal: Signal,
    bar: pandas.Series,
    shares: int,
) -> Tuple[Order, Order, Order]:
    """
    Create BUY + TP + SL orders at T+1 open for an ENTRY signal.

    Parameters
    ----------
    symbol : str
    signal : Signal
        ENTRY signal from strategy.
    bar : pandas.Series
        T+1 bar (open price used for fill).
    shares : int
        Number of shares.

    Returns
    -------
    Tuple[Order, Order, Order]
        (buy_order, tp_order, sl_order)
        BUY is already FILLED at bar open.
        TP and SL are PENDING.
    """
    entry_price = float(bar["open"])
    buy_order = Order(
        order_id=uuid.uuid4().hex[:12],
        order_type=OrderType.BUY,
        symbol=symbol,
        date_placed=bar.name,
        expected_fill=bar.name,
        quantity=shares,
        fill_date=bar.name,
        fill_price=entry_price,
        status=OrderStatus.FILLED,
    )

    tp_trigger: Optional[float] = None
    if signal.tp_pct is not None and signal.tp_pct > 0:
        tp_trigger = entry_price * (1 + signal.tp_pct)
    tp_order = Order(
        order_id=uuid.uuid4().hex[:12],
        order_type=OrderType.TAKE_PROFIT,
        symbol=symbol,
        date_placed=bar.name,
        expected_fill=bar.name,  # Unknown; evaluated bar by bar
        quantity=shares,
        trigger_price=tp_trigger,
        limit_price=tp_trigger,
        status=OrderStatus.PENDING,
        parent_order_id=buy_order.order_id,
    )

    sl_trigger: Optional[float] = None
    if signal.sl_pct is not None and signal.sl_pct > 0:
        sl_trigger = entry_price * (1 - signal.sl_pct)
    sl_order = Order(
        order_id=uuid.uuid4().hex[:12],
        order_type=OrderType.STOP_LOSS,
        symbol=symbol,
        date_placed=bar.name,
        expected_fill=bar.name,
        quantity=shares,
        trigger_price=sl_trigger,
        status=OrderStatus.PENDING,
        parent_order_id=buy_order.order_id,
    )

    return buy_order, tp_order, sl_order


def create_sell_order(
    symbol: str,
    bar: pandas.Series,
    quantity: int,
) -> Order:
    """
    Create a SELL market order at T+1 open for an EXIT signal.
    Filled at T+2 open (caller is responsible for placing on next bar).
    """
    return Order(
        order_id=uuid.uuid4().hex[:12],
        order_type=OrderType.SELL,
        symbol=symbol,
        date_placed=bar.name,
        expected_fill=bar.name,  # Will be filled next bar
        quantity=quantity,
        status=OrderStatus.PENDING,
    )
