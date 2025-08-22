"""Tests for strategy evaluation utilities."""
# TODO: review

import os
import sys
from pathlib import Path

import pandas
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from stock_indicator.strategy import evaluate_ema_sma_cross_strategy


def test_evaluate_ema_sma_cross_strategy_computes_win_rate(tmp_path: Path) -> None:
    price_values = [10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0]
    date_index = pandas.date_range("2020-01-01", periods=len(price_values), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": price_values,
            "close": price_values,
            "rs": [96.0] * len(price_values),
        }
    )
    csv_path = tmp_path / "test.csv"
    price_data_frame.to_csv(csv_path, index=False)

    total_trades, win_rate = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert total_trades == 1
    assert win_rate == 0.0


def test_evaluate_ema_sma_cross_strategy_normalizes_headers(tmp_path: Path) -> None:
    price_values = [10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0]
    date_index = pandas.date_range("2020-01-01", periods=len(price_values), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "OPEN": price_values,
            "CLOSE": price_values,
            "RS": [96.0] * len(price_values),
        }
    )
    csv_path = tmp_path / "test_uppercase.csv"
    price_data_frame.to_csv(csv_path, index=False)

    total_trades, win_rate = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert total_trades == 1
    assert win_rate == 0.0


def test_evaluate_ema_sma_cross_strategy_removes_ticker_suffix(tmp_path: Path) -> None:
    price_value_list = [10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0]
    date_index = pandas.date_range("2020-01-01", periods=len(price_value_list), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "Open RIV": price_value_list,
            "Close RIV": price_value_list,
            "RS": [96.0] * len(price_value_list),
        }
    )
    csv_path = tmp_path / "ticker_suffix.csv"
    price_data_frame.to_csv(csv_path, index=False)

    total_trades, win_rate = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert total_trades == 1
    assert win_rate == 0.0


def test_evaluate_ema_sma_cross_strategy_strips_leading_underscore(tmp_path: Path) -> None:
    price_value_list = [10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0]
    date_index = pandas.date_range("2020-01-01", periods=len(price_value_list), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "_Open RIV": price_value_list,
            "_Close RIV": price_value_list,
            "RS": [96.0] * len(price_value_list),
        }
    )
    csv_path = tmp_path / "leading_underscore.csv"
    price_data_frame.to_csv(csv_path, index=False)

    total_trades, win_rate = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert total_trades == 1
    assert win_rate == 0.0


def test_evaluate_ema_sma_cross_strategy_raises_value_error_for_missing_columns(
    tmp_path: Path,
) -> None:
    price_values = [10.0, 10.0, 10.0]
    date_index = pandas.date_range("2020-01-01", periods=len(price_values), freq="D")
    price_data_frame = pandas.DataFrame(
        {"Date": date_index, "Opening Price": price_values, "Closing Price": price_values}
    )
    csv_path = tmp_path / "test_missing.csv"
    price_data_frame.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)


def test_evaluate_ema_sma_cross_strategy_handles_multiindex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    price_value_list = [
        10.0,
        10.0,
        10.0,
        10.0,
        20.0,
        20.0,
        20.0,
        10.0,
        10.0,
        10.0,
    ]
    date_index = pandas.date_range("2020-01-01", periods=len(price_value_list), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            ("Date", ""): date_index,
            ("OPEN", "ignore"): price_value_list,
            ("CLOSE", "ignore"): price_value_list,
            ("RS", "ignore"): [96.0] * len(price_value_list),
        }
    )
    csv_path = tmp_path / "multi_index.csv"
    price_data_frame.to_csv(csv_path, index=False)

    original_read_csv = pandas.read_csv

    def patched_read_csv(*args: object, **kwargs: object) -> pandas.DataFrame:
        return original_read_csv(
            *args,
            header=[0, 1],
            index_col=0,
            parse_dates=[0],
            **{key: value for key, value in kwargs.items() if key not in {"header", "index_col", "parse_dates"}},
        )

    monkeypatch.setattr(pandas, "read_csv", patched_read_csv)
    total_trades, win_rate = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert total_trades == 1
    assert win_rate == 0.0


def test_evaluate_ema_sma_cross_strategy_requires_high_relative_strength(
    tmp_path: Path,
) -> None:
    price_values = [10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0]
    date_index = pandas.date_range("2020-01-01", periods=len(price_values), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": price_values,
            "close": price_values,
            "rs": [80.0] * len(price_values),
        }
    )
    csv_path = tmp_path / "low_rs.csv"
    price_data_frame.to_csv(csv_path, index=False)

    total_trades, win_rate = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert total_trades == 0
    assert win_rate == 0.0


def test_evaluate_ema_sma_cross_strategy_computes_missing_relative_strength(
    tmp_path: Path,
) -> None:
    price_values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    date_index = pandas.date_range("2020-01-01", periods=len(price_values), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": price_values,
            "close": price_values,
        }
    )
    csv_path = tmp_path / "missing_rs.csv"
    price_data_frame.to_csv(csv_path, index=False)

    total_trades, win_rate = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert (total_trades, win_rate) == (0, 0.0)
    updated_data_frame = pandas.read_csv(csv_path)
    assert "rs" in updated_data_frame.columns
