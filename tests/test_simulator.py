"""Tests for trade simulation utilities."""
# TODO: review

import os
import sys

import pandas

from stock_indicator.indicators import sma

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from stock_indicator.simulator import SimulationResult, simulate_trades


def test_simulate_trades_executes_trade_flow_with_default_column() -> None:
    price_data_frame = pandas.DataFrame(
        {"adj_close": [100.0, 102.0, 104.0, 103.0, 106.0]}
    )

    def entry_rule(current_row: pandas.Series) -> bool:
        return current_row["adj_close"] > 101.0

    def exit_rule(current_row: pandas.Series, entry_row: pandas.Series) -> bool:
        return current_row["adj_close"] > 105.0

    result = simulate_trades(price_data_frame, entry_rule, exit_rule)

    assert isinstance(result, SimulationResult)
    assert len(result.trades) == 1
    completed_trade = result.trades[0]
    expected_entry_date = price_data_frame.index[1]
    expected_exit_date = price_data_frame.index[4]
    assert completed_trade.entry_date == expected_entry_date
    assert completed_trade.exit_date == expected_exit_date
    assert completed_trade.entry_price == 102.0
    assert completed_trade.exit_price == 106.0
    assert completed_trade.profit == 4.0
    assert result.total_profit == 4.0


def test_simulate_trades_with_sma_strategy_uses_aligned_labels() -> None:
    """Verify SMA-based rules use matching index labels during comparison."""
    price_data_frame = pandas.DataFrame(
        {"close": [100.0, 102.0, 104.0, 103.0, 106.0]},
        index=[10, 11, 12, 13, 14],
    )
    simple_moving_average_series = sma(
        price_data_frame["close"], window_size=2
    )
    simple_moving_average_series = simple_moving_average_series.iloc[::-1]

    def entry_rule(current_row: pandas.Series) -> bool:
        """Determine when to enter a trade based on SMA."""
        row_label = current_row.name
        indicator_at_label = simple_moving_average_series.loc[row_label]
        if pandas.isna(indicator_at_label):
            return False
        return current_row["close"] > indicator_at_label

    def exit_rule(
        current_row: pandas.Series, entry_row: pandas.Series
    ) -> bool:
        """Determine when to exit a trade based on SMA."""
        row_label = current_row.name
        indicator_at_label = simple_moving_average_series.loc[row_label]
        if pandas.isna(indicator_at_label):
            return False
        return current_row["close"] < indicator_at_label

    result = simulate_trades(
        price_data_frame, entry_rule, exit_rule, price_column="close"
    )

    assert isinstance(result, SimulationResult)
    assert len(result.trades) == 1
    completed_trade = result.trades[0]
    expected_entry_date = price_data_frame.index[1]
    expected_exit_date = price_data_frame.index[3]
    assert completed_trade.entry_date == expected_entry_date
    assert completed_trade.exit_date == expected_exit_date
    assert completed_trade.entry_price == 102.0
    assert completed_trade.exit_price == 103.0
    assert completed_trade.profit == 1.0
    assert result.total_profit == 1.0
