"""Simulation utilities for executing trading strategies."""
# TODO: review

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import pandas


def calc_commission(shares: float, price: float) -> float:
    """Calculate the commission for a trade.

    Parameters
    ----------
    shares:
        Number of shares traded.
    price:
        Price per share.

    Returns
    -------
    float
        Commission charged for the trade side. The commission is ``0.0049``
        dollars per share with a minimum charge of ``0.99`` dollars and a
        maximum of ``0.5`` percent of the trade value.
    """
    basic_commission = shares * 0.0049
    commission_with_minimum = max(0.99, basic_commission)
    commission_cap = 0.005 * shares * price
    return min(commission_with_minimum, commission_cap)

@dataclass(frozen=True)
class Trade:
    """Record details for a completed trade."""

    entry_date: pandas.Timestamp
    exit_date: pandas.Timestamp
    entry_price: float
    exit_price: float
    profit: float
    holding_period: int


@dataclass
class SimulationResult:
    """Aggregate outcome of a trade simulation."""

    trades: List[Trade]
    total_profit: float


@dataclass
class OpenPosition:
    """Track details for an active portfolio position."""

    invested_amount: float
    share_count: float
    symbol: str | None = None
    last_known_price: float | None = None


def simulate_trades(
    data: pandas.DataFrame,
    entry_rule: Callable[[pandas.Series], bool],
    exit_rule: Callable[[pandas.Series, pandas.Series], bool],
    entry_price_column: str = "close",
    exit_price_column: str | None = None,
    stop_loss_percentage: float = 1.0,
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
    entry_price_column: str, default "close"
        Column name used for calculating entry price.
    exit_price_column: str, optional
        Column name used for calculating exit price. When ``None``,
        ``entry_price_column`` is used for both entry and exit prices.
    stop_loss_percentage: float, default 1.0
        Fractional loss from the entry price that triggers an exit on the next
        bar's opening price. Values greater than or equal to ``1.0`` disable
        the stop-loss mechanism.
    Commissions are calculated separately on entry and exit using
    :func:`calc_commission` and deducted from each trade's profit.

    Returns
    -------
    SimulationResult
        Contains the list of completed trades and aggregate profit.
    """
    trades: List[Trade] = []
    in_position = False
    entry_row: pandas.Series | None = None
    entry_row_index: int | None = None
    stop_loss_pending = False
    for row_index in range(len(data)):
        current_row = data.iloc[row_index]
        is_last_row = row_index == len(data) - 1
        if not in_position:
            if is_last_row:
                continue
            if entry_rule(current_row):
                in_position = True
                entry_row = current_row
                entry_row_index = row_index
        else:
            if entry_row is None or entry_row_index is None:
                continue
            if stop_loss_pending:
                entry_price = float(entry_row[entry_price_column])
                price_column_name = (
                    exit_price_column if exit_price_column is not None else entry_price_column
                )
                exit_price = float(current_row[price_column_name])
                entry_commission = calc_commission(1, entry_price)
                exit_commission = calc_commission(1, exit_price)
                profit_value = (
                    exit_price - entry_price - entry_commission - exit_commission
                )
                holding_period_value = row_index - entry_row_index
                trades.append(
                    Trade(
                        entry_date=data.index[entry_row_index],
                        exit_date=data.index[row_index],
                        entry_price=entry_price,
                        exit_price=exit_price,
                        profit=profit_value,
                        holding_period=holding_period_value,
                    )
                )
                in_position = False
                entry_row = None
                entry_row_index = None
                stop_loss_pending = False
                continue
            if exit_rule(current_row, entry_row):
                entry_price = float(entry_row[entry_price_column])
                price_column_name = (
                    exit_price_column if exit_price_column is not None else entry_price_column
                )
                exit_price = float(current_row[price_column_name])
                entry_commission = calc_commission(1, entry_price)
                exit_commission = calc_commission(1, exit_price)
                profit_value = (
                    exit_price - entry_price - entry_commission - exit_commission
                )
                holding_period_value = row_index - entry_row_index
                trades.append(
                    Trade(
                        entry_date=data.index[entry_row_index],
                        exit_date=data.index[row_index],
                        entry_price=entry_price,
                        exit_price=exit_price,
                        profit=profit_value,
                        holding_period=holding_period_value,
                    )
                )
                in_position = False
                entry_row = None
                entry_row_index = None
                continue
            if 0 < stop_loss_percentage < 1 and not is_last_row:
                entry_price = float(entry_row[entry_price_column])
                if float(current_row["close"]) <= entry_price * (1 - stop_loss_percentage):
                    stop_loss_pending = True
    if in_position and entry_row is not None and entry_row_index is not None:
        # TODO: review
        price_column_name = (
            exit_price_column if exit_price_column is not None else entry_price_column
        )
        final_row = data.iloc[-1]
        entry_price = float(entry_row[entry_price_column])
        exit_price = float(final_row[price_column_name])
        entry_commission = calc_commission(1, entry_price)
        exit_commission = calc_commission(1, exit_price)
        profit_value = exit_price - entry_price - entry_commission - exit_commission
        holding_period_value = len(data) - entry_row_index - 1
        trades.append(
            Trade(
                entry_date=data.index[entry_row_index],
                exit_date=data.index[-1],
                entry_price=entry_price,
                exit_price=exit_price,
                profit=profit_value,
                holding_period=holding_period_value,
            )
        )
    total_profit = sum(completed_trade.profit for completed_trade in trades)
    return SimulationResult(trades=trades, total_profit=total_profit)


# TODO: review
def calculate_maximum_concurrent_positions(
    simulation_results: Iterable[SimulationResult],
) -> int:
    """Determine the highest number of simultaneous open positions.

    Parameters
    ----------
    simulation_results : Iterable[SimulationResult]
        Collection of simulation outputs containing trades to evaluate.

    Returns
    -------
    int
        Maximum count of concurrent positions across all simulations.
    """
    events: List[tuple[pandas.Timestamp, int]] = []
    for simulation_result in simulation_results:
        for current_trade in simulation_result.trades:
            events.append((current_trade.entry_date, 1))
            events.append((current_trade.exit_date, -1))
    events.sort(key=lambda event: (event[0], event[1]))
    open_position_count = 0
    maximum_open_position_count = 0
    for _, change in events:
        open_position_count += change
        if open_position_count > maximum_open_position_count:
            maximum_open_position_count = open_position_count
    return maximum_open_position_count
# TODO: review
def simulate_portfolio_balance(
    trades: Iterable[Trade],
    starting_cash: float,
    eligible_symbol_count: int | Dict[pandas.Timestamp, int],
    withdraw_amount: float = 0.0,
) -> float:
    """Simulate capital allocation across multiple trades.

    Parameters
    ----------
    trades : Iterable[Trade]
        Collection of trades containing entry and exit information.
    starting_cash : float
        Initial cash available for trading.
    eligible_symbol_count : int | Dict[pandas.Timestamp, int]
        Total number of symbols considered for trading. When provided as a
        dictionary, the key is the trading date and the value represents the
        number of eligible symbols on that day. This mirrors the values
        produced by :func:`stock_indicator.volume.count_symbols_with_average_dollar_volume_above`.
        Entry and exit commissions are deducted using :func:`calc_commission`.
    withdraw_amount : float, optional
        Cash amount removed from the balance at the end of each calendar year.
        Defaults to ``0.0``.

    Returns
    -------
    float
        Cash balance after all trades have been executed.
    """
    events: List[tuple[pandas.Timestamp, int, Trade]] = []
    for trade in trades:
        events.append((trade.entry_date, 1, trade))
        events.append((trade.exit_date, 0, trade))
    events.sort(key=lambda event_tuple: (event_tuple[0], event_tuple[1]))
    cash_balance = starting_cash
    open_trades: dict[Trade, OpenPosition] = {}
    current_year: int | None = None

    def get_symbol_count(current_date: pandas.Timestamp) -> int:
        if isinstance(eligible_symbol_count, dict):
            return eligible_symbol_count.get(current_date, 0)
        return eligible_symbol_count

    for event_timestamp, event_type, trade in events:
        if current_year is None:
            current_year = event_timestamp.year
        while event_timestamp.year > current_year:
            cash_balance -= withdraw_amount
            current_year += 1
        if event_type == 0:
            if trade in open_trades:
                position_details = open_trades.pop(trade)
                proceeds = position_details.share_count * trade.exit_price
                exit_commission = calc_commission(
                    position_details.share_count, trade.exit_price
                )
                cash_balance += proceeds - exit_commission
        else:
            remaining_slots = get_symbol_count(event_timestamp) - len(open_trades)
            if remaining_slots <= 0 or cash_balance <= 0:
                continue
            budget_per_position = cash_balance / remaining_slots
            share_count = math.floor(budget_per_position / trade.entry_price)
            if share_count <= 0:
                continue
            invested_amount = share_count * trade.entry_price
            if cash_balance - invested_amount >= trade.entry_price:
                share_count += 1
                invested_amount = share_count * trade.entry_price
            entry_commission = calc_commission(share_count, trade.entry_price)
            open_trades[trade] = OpenPosition(
                invested_amount=invested_amount, share_count=share_count
            )
            cash_balance -= invested_amount + entry_commission
    if current_year is not None:
        cash_balance -= withdraw_amount
    return cash_balance


def calculate_max_drawdown(
    trades: Iterable[Trade],
    starting_cash: float,
    eligible_symbol_counts_by_date: int | Dict[pandas.Timestamp, int],
    trade_symbol_lookup: Dict[Trade, str],
    closing_price_series_by_symbol: Dict[str, pandas.Series],
    withdraw_amount: float = 0.0,
) -> float:
    """Compute the maximum portfolio drawdown across the simulation period.

    The function simulates the portfolio balance for each calendar day using
    the same allocation logic as :func:`simulate_portfolio_balance`. It tracks
    the running peak balance and records the greatest percentage decline from
    that peak. The drawdown is expressed as a decimal where ``0.25`` denotes a
    twenty-five percent drop.

    Parameters
    ----------
    trades: Iterable[Trade]
        Collection of trades with entry and exit information.
    starting_cash: float
        Initial cash available for trading.
    eligible_symbol_counts_by_date: int | Dict[pandas.Timestamp, int]
        Total number of symbols considered for trading. When provided as a
        dictionary, keys are trading dates and values represent the counts of
        eligible symbols on those dates. This mirrors the values produced by
        :func:`stock_indicator.volume.count_symbols_with_average_dollar_volume_above`.
    trade_symbol_lookup: Dict[Trade, str]
        Mapping from each trade to its associated trading symbol.
    closing_price_series_by_symbol: Dict[str, pandas.Series]
        Mapping of symbol to a series of daily closing prices used to revalue
        open positions.
    withdraw_amount: float, optional
        Cash amount removed from the balance at the end of each calendar year.
        Defaults to ``0.0``.

    Returns
    -------
    float
        Largest fractional decline from any previous portfolio peak.
    """

    events: List[tuple[pandas.Timestamp, int, Trade]] = []
    for current_trade in trades:
        events.append((current_trade.entry_date, 1, current_trade))
        events.append((current_trade.exit_date, 0, current_trade))
    events.sort(key=lambda event_tuple: (event_tuple[0], event_tuple[1]))
    if not events:
        return 0.0

    def get_symbol_count(current_date: pandas.Timestamp) -> int:
        if isinstance(eligible_symbol_counts_by_date, dict):
            return eligible_symbol_counts_by_date.get(current_date, 0)
        return eligible_symbol_counts_by_date

    cash_balance = starting_cash
    open_trades: dict[Trade, OpenPosition] = {}
    maximum_portfolio_value = starting_cash
    maximum_drawdown_value = 0.0
    current_year = events[0][0].year

    start_date = events[0][0]
    end_date = events[-1][0]
    event_index = 0
    current_date = start_date

    while current_date <= end_date:
        while event_index < len(events) and events[event_index][0] == current_date:
            _, event_type, trade = events[event_index]
            symbol_name = trade_symbol_lookup.get(trade, "")
            if event_type == 0:
                if trade in open_trades:
                    position_details = open_trades.pop(trade)
                    proceeds = position_details.share_count * trade.exit_price
                    exit_commission = calc_commission(
                        position_details.share_count, trade.exit_price
                    )
                    cash_balance += proceeds - exit_commission
            else:
                remaining_slots = get_symbol_count(current_date) - len(open_trades)
                if remaining_slots > 0 and cash_balance > 0:
                    budget_per_position = cash_balance / remaining_slots
                    share_count = math.floor(budget_per_position / trade.entry_price)
                    if share_count > 0:
                        invested_amount = share_count * trade.entry_price
                        if cash_balance - invested_amount >= trade.entry_price:
                            share_count += 1
                            invested_amount = share_count * trade.entry_price
                        entry_commission = calc_commission(share_count, trade.entry_price)
                        open_trades[trade] = OpenPosition(
                            invested_amount=invested_amount,
                            share_count=share_count,
                            symbol=symbol_name,
                            last_known_price=trade.entry_price,
                        )
                        cash_balance -= invested_amount + entry_commission
            event_index += 1

        for position_details in open_trades.values():
            price_series = closing_price_series_by_symbol.get(position_details.symbol)
            if price_series is not None and current_date in price_series.index:
                position_details.last_known_price = float(price_series.loc[current_date])

        portfolio_value = cash_balance + sum(
            position_details.share_count * (position_details.last_known_price or 0.0)
            for position_details in open_trades.values()
        )
        if portfolio_value > maximum_portfolio_value:
            maximum_portfolio_value = portfolio_value
        elif maximum_portfolio_value > 0:
            drawdown = (maximum_portfolio_value - portfolio_value) / maximum_portfolio_value
            if drawdown > maximum_drawdown_value:
                maximum_drawdown_value = drawdown

        next_day = current_date + pandas.Timedelta(days=1)
        if next_day.year > current_year:
            cash_balance -= withdraw_amount
            maximum_portfolio_value = max(0.0, maximum_portfolio_value - withdraw_amount)
            portfolio_value = cash_balance + sum(
                position_details.share_count * (position_details.last_known_price or 0.0)
                for position_details in open_trades.values()
            )
            if portfolio_value > maximum_portfolio_value:
                maximum_portfolio_value = portfolio_value
            elif maximum_portfolio_value > 0:
                drawdown = (
                    (maximum_portfolio_value - portfolio_value) / maximum_portfolio_value
                )
                if drawdown > maximum_drawdown_value:
                    maximum_drawdown_value = drawdown
            current_year = next_day.year

        current_date = next_day

    return maximum_drawdown_value


def calculate_annual_returns(
    trades: Iterable[Trade],
    starting_cash: float,
    eligible_symbol_count: int | Dict[pandas.Timestamp, int],
    simulation_start: pandas.Timestamp,
    withdraw_amount: float = 0.0,
) -> Dict[int, float]:
    """Compute yearly portfolio returns for a series of trades.

    The portfolio balance is simulated using the same allocation logic as
    :func:`simulate_portfolio_balance`. The function records the portfolio
    value at the start and end of each calendar year and calculates the return
    for that year.

    Parameters
    ----------
    trades: Iterable[Trade]
        Collection of trades containing entry and exit information.
    starting_cash: float
        Initial cash available for trading.
    eligible_symbol_count: int | Dict[pandas.Timestamp, int]
        Total number of symbols considered for trading. When provided as a
        dictionary, keys are trading dates and values are the counts of
        eligible symbols on those dates. This mirrors the values produced by
        :func:`stock_indicator.volume.count_symbols_with_average_dollar_volume_above`.
    simulation_start: pandas.Timestamp
        Timestamp indicating the first day of the simulation. Years prior to
        the first trade will be emitted with a zero return.
    withdraw_amount: float, optional
        Cash amount removed from the balance at the end of each calendar year.
        Defaults to ``0.0``.

    Returns
    -------
    Dict[int, float]
        Mapping of year to return percentage for that year. The value is
        expressed as a decimal where ``0.10`` represents a ten percent return.
    """  # TODO: review
    events: List[tuple[pandas.Timestamp, int, Trade]] = []
    for completed_trade in trades:
        events.append((completed_trade.entry_date, 1, completed_trade))
        events.append((completed_trade.exit_date, 0, completed_trade))
    events.sort(key=lambda event_tuple: (event_tuple[0], event_tuple[1]))

    cash_balance = starting_cash
    open_trades: dict[Trade, OpenPosition] = {}
    annual_returns: Dict[int, float] = {}

    current_year = simulation_start.year
    year_start_value = cash_balance

    def get_symbol_count(current_date: pandas.Timestamp) -> int:
        if isinstance(eligible_symbol_count, dict):
            return eligible_symbol_count.get(current_date, 0)
        return eligible_symbol_count

    for event_timestamp, event_type, completed_trade in events:
        while event_timestamp.year > current_year:
            year_end_value = cash_balance + sum(
                position_details.invested_amount for position_details in open_trades.values()
            )
            if year_start_value == 0:
                annual_returns[current_year] = 0.0
            else:
                annual_returns[current_year] = (
                    (year_end_value - year_start_value) / year_start_value
                )
            cash_balance -= withdraw_amount
            year_start_value = cash_balance + sum(
                position_details.invested_amount for position_details in open_trades.values()
            )
            current_year += 1

        if event_type == 0:
            if completed_trade in open_trades:
                position_details = open_trades.pop(completed_trade)
                proceeds = position_details.share_count * completed_trade.exit_price
                exit_commission = calc_commission(
                    position_details.share_count, completed_trade.exit_price
                )
                cash_balance += proceeds - exit_commission
        else:
            remaining_slots = get_symbol_count(event_timestamp) - len(open_trades)
            if remaining_slots <= 0 or cash_balance <= 0:
                continue
            budget_per_position = cash_balance / remaining_slots
            share_count = math.floor(
                budget_per_position / completed_trade.entry_price
            )
            if share_count <= 0:
                continue
            invested_amount = share_count * completed_trade.entry_price
            if cash_balance - invested_amount >= completed_trade.entry_price:
                share_count += 1
                invested_amount = share_count * completed_trade.entry_price
            entry_commission = calc_commission(share_count, completed_trade.entry_price)
            open_trades[completed_trade] = OpenPosition(
                invested_amount=invested_amount, share_count=share_count
            )
            cash_balance -= invested_amount + entry_commission

    year_end_value = cash_balance + sum(
        position_details.invested_amount for position_details in open_trades.values()
    )
    if year_start_value == 0:
        annual_returns[current_year] = 0.0
    else:
        annual_returns[current_year] = (
            (year_end_value - year_start_value) / year_start_value
        )
    cash_balance -= withdraw_amount

    return annual_returns


def calculate_annual_trade_counts(trades: Iterable[Trade]) -> Dict[int, int]:
    """Count completed trades for each calendar year.

    Parameters
    ----------
    trades:
        Collection of trades containing exit dates.

    Returns
    -------
    Dict[int, int]
        Mapping of year to number of trades that closed during that year.
    """
    trade_counts: Dict[int, int] = {}
    for completed_trade in trades:
        year_value = completed_trade.exit_date.year
        if year_value in trade_counts:
            trade_counts[year_value] += 1
        else:
            trade_counts[year_value] = 1
    return trade_counts
