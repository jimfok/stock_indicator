"""Tests for strategy evaluation utilities."""
# TODO: review

import os
import sys
from pathlib import Path

import pandas

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from stock_indicator.strategy import evaluate_ema_sma_cross_strategy


def test_evaluate_ema_sma_cross_strategy_computes_win_rate(tmp_path: Path) -> None:
    price_values = [10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0]
    date_index = pandas.date_range("2020-01-01", periods=len(price_values), freq="D")
    price_data_frame = pandas.DataFrame(
        {"Date": date_index, "open": price_values, "close": price_values}
    )
    csv_path = tmp_path / "test.csv"
    price_data_frame.to_csv(csv_path, index=False)

    total_trades, win_rate = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert total_trades == 1
    assert win_rate == 0.0
