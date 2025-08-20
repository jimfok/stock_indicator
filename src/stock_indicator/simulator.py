"""Simulation utilities for executing trading strategies."""
# TODO: review

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import pandas


@dataclass
class Trade:
    """Record details for a completed trade."""

    entry_date: pandas.Timestamp
    exit_date: pandas.Timestamp
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
    entry_price_column: str = "adj_close",
    exit_price_column: str | None = None,
) -> SimulationResult:
    """Simulate trades using supplied entry and exit rules.

    Parameters
    ----------
    data: pandas.DataFrame
        Data frame containing the price data.
    entry_rule: Callable[[pandas.Series], bool]
        Function invoked for each row to determine trade entry.
    exit_rule: Callable[[pandas.Series, pandas.Series], bool]
        Function invoked with the current row and the entry row to determine
        when to close the trade.
    entry_price_column: str, default "adj_close"
        Column name used for calculating entry price.
    exit_price_column: str, optional
        Column name used for calculating exit price. When ``None``,
        ``entry_price_column`` is used for both entry and exit prices.

    Returns
    -------
    SimulationResult
        Contains the list of completed trades and aggregate profit.
    """
    trades: List[Trade] = []
    in_position = False
    entry_row: pandas.Series | None = None
    entry_row_index: int | None = None
    for row_index in range(len(data)):
        current_row = data.iloc[row_index]
        if not in_position:
            if entry_rule(current_row):
                in_position = True
                entry_row = current_row
                entry_row_index = row_index
        else:
            if entry_row is None or entry_row_index is None:
                continue
            if exit_rule(current_row, entry_row):
                entry_price = float(entry_row[entry_price_column])
                price_column_name = (
                    exit_price_column if exit_price_column is not None else entry_price_column
                )
                exit_price = float(current_row[price_column_name])
                profit_value = exit_price - entry_price
                trades.append(
                    Trade(
                        entry_date=data.index[entry_row_index],
                        exit_date=data.index[row_index],
                        entry_price=entry_price,
                        exit_price=exit_price,
                        profit=profit_value,
                    )
                )
                in_position = False
                entry_row = None
                entry_row_index = None
    total_profit = sum(completed_trade.profit for completed_trade in trades)
    return SimulationResult(trades=trades, total_profit=total_profit)
