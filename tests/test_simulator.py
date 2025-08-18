"""Tests for trade simulation utilities."""
# TODO: review

import os
import sys

import pandas

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from stock_indicator.simulator import SimulationResult, simulate_trades


def test_simulate_trades_executes_trade_flow() -> None:
    data = pandas.DataFrame({"close": [100.0, 102.0, 104.0, 103.0, 106.0]})

    def entry_rule(current_row: pandas.Series) -> bool:
        return current_row["close"] > 101.0

    def exit_rule(current_row: pandas.Series, entry_row: pandas.Series) -> bool:
        return current_row["close"] > 105.0

    result = simulate_trades(data, entry_rule, exit_rule)

    assert isinstance(result, SimulationResult)
    assert len(result.trades) == 1
    completed_trade = result.trades[0]
    assert completed_trade.entry_index == 1
    assert completed_trade.exit_index == 4
    assert completed_trade.entry_price == 102.0
    assert completed_trade.exit_price == 106.0
    assert completed_trade.profit == 4.0
    assert result.total_profit == 4.0
