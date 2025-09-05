"""Tests for loading price data with both Close and Adj Close columns."""
# TODO: review

from pathlib import Path
import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import stock_indicator.strategy as strategy_module
from stock_indicator.indicators import sma


def test_load_price_data_handles_adj_close_column() -> None:
    """Data frame should have unique columns and allow SMA assignment."""
    csv_file_path = Path(__file__).resolve().parent / "fixtures" / "sample_prices.csv"
    price_data_frame = strategy_module.load_price_data(csv_file_path)
    assert price_data_frame.columns.is_unique
    price_data_frame["adj_close_sma"] = sma(price_data_frame["adj_close"], 2)
    expected_last_value = (10.7 + 10.8) / 2
    assert price_data_frame.loc["2024-01-03", "adj_close_sma"] == pytest.approx(expected_last_value)
