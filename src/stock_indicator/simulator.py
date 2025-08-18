"""Simulation utilities for executing trading strategies."""
# TODO: review

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import pandas


@dataclass
class Trade:
    """Record details for a completed trade."""

    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    profit: float


@dataclass
class SimulationResult:
    """Aggregate outcome of a trade simulation."""

    trades: List[Trade]
    total_profit: float


def simulate_trades(
    data: pandas.DataFrame,
    entry_rule: Callable[[pandas.Series], bool],
    exit_rule: Callable[[pandas.Series, pandas.Series], bool],
) -> SimulationResult:
    """Simulate trades using supplied entry and exit rules.

    Parameters
    ----------
    data: pandas.DataFrame
        Data frame containing at least a ``close`` column.
    entry_rule: Callable[[pandas.Series], bool]
        Function invoked for each row to determine trade entry.
    exit_rule: Callable[[pandas.Series, pandas.Series], bool]
        Function invoked with the current row and the entry row to determine
        when to close the trade.

    Returns
    -------
    SimulationResult
        Contains the list of completed trades and aggregate profit.
    """
    trades: List[Trade] = []
    in_position = False
    entry_row: pandas.Series | None = None
    entry_index: int | None = None
    for row_index in range(len(data)):
        current_row = data.iloc[row_index]
        if not in_position:
            if entry_rule(current_row):
                in_position = True
                entry_row = current_row
                entry_index = row_index
        else:
            if entry_row is None or entry_index is None:
                continue
            if exit_rule(current_row, entry_row):
                entry_price = float(entry_row["close"])
                exit_price = float(current_row["close"])
                profit_value = exit_price - entry_price
                trades.append(
                    Trade(
                        entry_index=entry_index,
                        exit_index=row_index,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        profit=profit_value,
                    )
                )
                in_position = False
                entry_row = None
                entry_index = None
    total_profit = sum(completed_trade.profit for completed_trade in trades)
    return SimulationResult(trades=trades, total_profit=total_profit)
