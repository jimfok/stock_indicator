"""Tests for strategy evaluation utilities."""
# TODO: review

import os
import sys
import datetime
from pathlib import Path
from typing import Iterable

import pandas
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import stock_indicator.strategy as strategy
from stock_indicator.simulator import SimulationResult, Trade, calc_commission
from stock_indicator.chip_filter import calculate_chip_concentration_metrics

from stock_indicator.strategy import (
    evaluate_ema_sma_cross_strategy,
    evaluate_kalman_channel_strategy,
    evaluate_combined_strategy,
    parse_strategy_name,
)


def test_evaluate_ema_sma_cross_strategy_computes_win_rate(tmp_path: Path) -> None:
    initial_price_values = [10.0] * 150
    pattern_price_values = [
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
    price_values = initial_price_values + pattern_price_values
    date_index = pandas.date_range("2020-01-01", periods=len(price_values), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": price_values,
            "close": price_values,
        }
    )
    csv_path = tmp_path / "test.csv"
    price_data_frame.to_csv(csv_path, index=False)

    result = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert result.total_trades == 1
    assert result.win_rate == 0.0


def test_evaluate_ema_sma_cross_strategy_normalizes_headers(tmp_path: Path) -> None:
    initial_price_values = [10.0] * 150
    pattern_price_values = [
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
    price_values = initial_price_values + pattern_price_values
    date_index = pandas.date_range("2020-01-01", periods=len(price_values), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "OPEN": price_values,
            "CLOSE": price_values,
        }
    )
    csv_path = tmp_path / "test_uppercase.csv"
    price_data_frame.to_csv(csv_path, index=False)

    result = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert result.total_trades == 1
    assert result.win_rate == 0.0


def test_evaluate_ema_sma_cross_strategy_removes_ticker_suffix(tmp_path: Path) -> None:
    initial_price_values = [10.0] * 150
    pattern_price_values = [
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
    price_value_list = initial_price_values + pattern_price_values
    date_index = pandas.date_range("2020-01-01", periods=len(price_value_list), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "Open RIV": price_value_list,
            "Close RIV": price_value_list,
        }
    )
    csv_path = tmp_path / "ticker_suffix.csv"
    price_data_frame.to_csv(csv_path, index=False)

    result = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert result.total_trades == 1
    assert result.win_rate == 0.0


def test_evaluate_ema_sma_cross_strategy_strips_leading_underscore(
    tmp_path: Path,
) -> None:
    initial_price_values = [10.0] * 150
    pattern_price_values = [
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
    price_value_list = initial_price_values + pattern_price_values
    date_index = pandas.date_range("2020-01-01", periods=len(price_value_list), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "_Open RIV": price_value_list,
            "_Close RIV": price_value_list,
        }
    )
    csv_path = tmp_path / "leading_underscore.csv"
    price_data_frame.to_csv(csv_path, index=False)

    result = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert result.total_trades == 1
    assert result.win_rate == 0.0


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
    initial_price_values = [10.0] * 150
    pattern_price_values = [
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
    price_value_list = initial_price_values + pattern_price_values
    date_index = pandas.date_range("2020-01-01", periods=len(price_value_list), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            ("Date", ""): date_index,
            ("OPEN", "ignore"): price_value_list,
            ("CLOSE", "ignore"): price_value_list,
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
    result = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert result.total_trades == 1
    assert result.win_rate == 0.0


def test_evaluate_ema_sma_cross_strategy_requires_close_above_long_term_sma(
    tmp_path: Path,
) -> None:
    initial_price_values = [20.0] * 150
    pattern_price_values = [
        20.0,
        20.0,
        20.0,
        20.0,
        10.0,
        10.0,
        10.0,
        20.0,
        20.0,
        20.0,
    ]
    price_values = initial_price_values + pattern_price_values
    date_index = pandas.date_range("2020-01-01", periods=len(price_values), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": price_values,
            "close": price_values,
        }
    )
    csv_path = tmp_path / "below_long_sma.csv"
    price_data_frame.to_csv(csv_path, index=False)

    result = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert result.total_trades == 0
    assert result.win_rate == 0.0


def test_evaluate_ema_sma_cross_strategy_ignores_missing_relative_strength(
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

    result = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert (result.total_trades, result.win_rate) == (0, 0.0)
    updated_data_frame = pandas.read_csv(csv_path)
    assert "rs" not in updated_data_frame.columns


def test_evaluate_ema_sma_cross_strategy_computes_profit_and_loss_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    price_values = [10.0] * 160
    date_index = pandas.date_range("2020-01-01", periods=len(price_values), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": price_values,
            "close": price_values,
        }
    )
    csv_path = tmp_path / "statistics.csv"
    price_data_frame.to_csv(csv_path, index=False)

    trades = [
        Trade(
            entry_date=date_index[0],
            exit_date=date_index[1],
            entry_price=10.0,
            exit_price=12.0,
            profit=2.0,
            holding_period=1,
        ),
        Trade(
            entry_date=date_index[2],
            exit_date=date_index[3],
            entry_price=10.0,
            exit_price=9.0,
            profit=-1.0,
            holding_period=1,
        ),
    ]
    simulation_result = SimulationResult(trades=trades, total_profit=1.0)

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        return simulation_result

    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)

    result = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert result.total_trades == 2
    assert result.win_rate == 0.5
    assert result.mean_profit_percentage == pytest.approx(0.2)
    assert result.profit_percentage_standard_deviation == 0.0
    assert result.mean_loss_percentage == pytest.approx(0.1)
    assert result.loss_percentage_standard_deviation == 0.0
    assert result.mean_holding_period == pytest.approx(1.0)
    assert result.holding_period_standard_deviation == 0.0


def test_evaluate_kalman_channel_strategy_generates_trade(tmp_path: Path) -> None:
    initial_price_values = [20.0] * 20
    pattern_price_values = [
        10.0,
        20.0,
        20.0,
        20.0,
        10.0,
        10.0,
    ]
    price_values = initial_price_values + pattern_price_values
    date_index = pandas.date_range("2020-01-01", periods=len(price_values), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": price_values,
            "close": price_values,
        }
    )
    csv_path = tmp_path / "kalman.csv"
    price_data_frame.to_csv(csv_path, index=False)

    result = evaluate_kalman_channel_strategy(tmp_path)

    assert result.total_trades == 1
    assert result.win_rate == 0.0


def test_evaluate_combined_strategy_different_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """evaluate_combined_strategy should aggregate results for mixed strategies."""
    price_values = [10.0] * 160
    date_index = pandas.date_range("2020-01-01", periods=len(price_values), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": price_values,
            "close": price_values,
        }
    )
    csv_path = tmp_path / "combined.csv"
    price_data_frame.to_csv(csv_path, index=False)

    trades = [
        Trade(
            entry_date=date_index[0],
            exit_date=date_index[1],
            entry_price=10.0,
            exit_price=12.0,
            profit=2.0,
            holding_period=1,
        ),
        Trade(
            entry_date=date_index[2],
            exit_date=date_index[3],
            entry_price=10.0,
            exit_price=9.0,
            profit=-1.0,
            holding_period=1,
        ),
    ]
    simulation_result = SimulationResult(trades=trades, total_profit=1.0)

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        return simulation_result

    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)

    result = evaluate_combined_strategy(
        tmp_path, "ema_sma_cross", "kalman_filtering"
    )

    assert result.total_trades == 2
    assert result.win_rate == 0.5


def test_evaluate_combined_strategy_calculates_compound_annual_growth_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function should compute CAGR based on final balance and duration."""

    date_index = pandas.date_range("2020-01-01", periods=370, freq="D")
    price_data_frame = pandas.DataFrame(
        {"Date": date_index, "open": [10.0] * 370, "close": [10.0] * 370}
    )
    csv_path = tmp_path / "cagr_test.csv"
    price_data_frame.to_csv(csv_path, index=False)

    trade = Trade(
        entry_date=date_index[0],
        exit_date=date_index[365],
        entry_price=10.0,
        exit_price=11.0,
        profit=1.0,
        holding_period=365,
    )
    simulation_result = SimulationResult(trades=[trade], total_profit=1.0)

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        return simulation_result

    def fake_simulate_portfolio_balance(*args: object, **kwargs: object) -> float:
        return 3300.0

    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)
    monkeypatch.setattr(
        strategy, "simulate_portfolio_balance", fake_simulate_portfolio_balance
    )
    monkeypatch.setattr(strategy, "calculate_annual_returns", lambda *a, **k: {})
    monkeypatch.setattr(strategy, "calculate_annual_trade_counts", lambda *a, **k: {})

    result = evaluate_combined_strategy(tmp_path, "ema_sma_cross", "ema_sma_cross")

    duration_years = (trade.exit_date - trade.entry_date).days / 365.25
    expected_growth_rate = (3300.0 / 3000.0) ** (1 / duration_years) - 1
    assert result.compound_annual_growth_rate == pytest.approx(
        expected_growth_rate
    )


def test_evaluate_combined_strategy_reports_max_drawdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """evaluate_combined_strategy should report the largest portfolio decline."""
    date_index = pandas.date_range("2020-01-01", periods=200, freq="D")
    closing_prices = [10.0] * 200
    closing_prices[1] = 8.0
    closing_prices[2] = 12.0
    price_data_frame = pandas.DataFrame(
        {"Date": date_index, "open": [10.0] * 200, "close": closing_prices}
    )
    csv_path = tmp_path / "AAA.csv"
    price_data_frame.to_csv(csv_path, index=False)

    trade = Trade(
        entry_date=pandas.Timestamp("2020-01-01"),
        exit_date=pandas.Timestamp("2020-01-03"),
        entry_price=10.0,
        exit_price=12.0,
        profit=2.0 - calc_commission(1, 10.0) - calc_commission(1, 12.0),
        holding_period=2,
    )
    simulation_result = SimulationResult(trades=[trade], total_profit=trade.profit)

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        return simulation_result

    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)
    monkeypatch.setattr(strategy, "calculate_annual_returns", lambda *a, **k: {})
    monkeypatch.setattr(strategy, "calculate_annual_trade_counts", lambda *a, **k: {})
    monkeypatch.setattr(strategy, "simulate_portfolio_balance", lambda *a, **k: 1000.0)

    result = evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross",
        "ema_sma_cross",
        starting_cash=1000.0,
        maximum_position_count=1,
    )

    entry_commission = calc_commission(100, 10.0)
    cash_after_entry = 1000.0 - 100 * 10.0 - entry_commission
    lowest_portfolio_value = cash_after_entry + 100 * 8.0
    expected_drawdown = (1000.0 - lowest_portfolio_value) / 1000.0
    assert result.maximum_drawdown == pytest.approx(expected_drawdown)


def test_evaluate_combined_strategy_unsupported_name(tmp_path: Path) -> None:
    """evaluate_combined_strategy should raise for unknown strategies."""
    with pytest.raises(ValueError, match="Unsupported strategy"):
        evaluate_combined_strategy(tmp_path, "unknown", "ema_sma_cross")


def test_evaluate_combined_strategy_rejects_sell_only_buy(tmp_path: Path) -> None:
    """evaluate_combined_strategy should reject sell-only strategies used for buying."""
    with pytest.raises(ValueError, match="Unsupported strategy"):
        evaluate_combined_strategy(tmp_path, "kalman_filtering", "ema_sma_cross")


def test_evaluate_combined_strategy_passes_window_size_and_renames_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function should pass the window size and append it to signal names."""

    date_index = pandas.date_range("2020-01-01", periods=2, freq="D")
    price_data_frame = pandas.DataFrame(
        {"Date": date_index, "open": [10.0, 10.0], "close": [10.0, 10.0]}
    )
    csv_path = tmp_path / "window.csv"
    price_data_frame.to_csv(csv_path, index=False)

    captured_arguments: dict[str, int | None] = {"window_size": None}
    captured_column_names: list[str] = []

    def fake_attach_signals(frame: pandas.DataFrame, window_size: int = 50) -> None:
        captured_arguments["window_size"] = window_size
        frame["ema_sma_cross_with_slope_entry_signal"] = [True, False]
        frame["ema_sma_cross_with_slope_exit_signal"] = [False, True]

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        passed_frame: pandas.DataFrame = kwargs["data"]
        captured_column_names.extend(passed_frame.columns)
        trade = Trade(
            entry_date=date_index[0],
            exit_date=date_index[1],
            entry_price=10.0,
            exit_price=10.0,
            profit=0.0,
            holding_period=1,
        )
        return SimulationResult(trades=[trade], total_profit=0.0)

    monkeypatch.setattr(
        strategy, "attach_ema_sma_cross_with_slope_signals", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.BUY_STRATEGIES, "ema_sma_cross_with_slope", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.SELL_STRATEGIES, "ema_sma_cross_with_slope", fake_attach_signals
    )
    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)

    try:
        evaluate_combined_strategy(
            tmp_path, "ema_sma_cross_with_slope_40", "ema_sma_cross_with_slope_40"
        )
    except ValueError as error:
        pytest.fail(f"Unexpected ValueError: {error}")

    assert captured_arguments["window_size"] == 40
    assert "ema_sma_cross_with_slope_40_entry_signal" in captured_column_names
    assert "ema_sma_cross_with_slope_40_exit_signal" in captured_column_names


def test_evaluate_combined_strategy_passes_angle_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function should pass the angle range to the strategy function."""

    date_index = pandas.date_range("2020-01-01", periods=2, freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": [10.0, 10.0],
            "close": [10.0, 10.0],
            "volume": [1.0, 1.0],
        }
    )
    csv_path = tmp_path / "slope.csv"
    price_data_frame.to_csv(csv_path, index=False)

    captured_arguments: dict[str, tuple[float, float] | None] = {
        "angle_range": None
    }

    def fake_attach_signals(
        frame: pandas.DataFrame,
        window_size: int = 40,
        angle_range: tuple[float, float] = (
            -16.69924423399362,
            64.95379922035721,
        ),
    ) -> None:
        captured_arguments["angle_range"] = angle_range
        frame["ema_sma_cross_with_slope_entry_signal"] = [True, False]
        frame["ema_sma_cross_with_slope_exit_signal"] = [False, True]

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        trade = Trade(
            entry_date=date_index[0],
            exit_date=date_index[1],
            entry_price=10.0,
            exit_price=10.0,
            profit=0.0,
            holding_period=1,
        )
        return SimulationResult(trades=[trade], total_profit=0.0)

    monkeypatch.setattr(
        strategy, "attach_ema_sma_cross_with_slope_signals", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.BUY_STRATEGIES, "ema_sma_cross_with_slope", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.SELL_STRATEGIES, "ema_sma_cross_with_slope", fake_attach_signals
    )
    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)

    evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross_with_slope_-26.6_26.6",
        "ema_sma_cross_with_slope_-26.6_26.6",
    )

    assert captured_arguments["angle_range"] == (-26.6, 26.6)


def test_evaluate_combined_strategy_passes_angle_range_with_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function should pass slope range for strategies using volume."""

    date_index = pandas.date_range("2020-01-01", periods=2, freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": [10.0, 10.0],
            "close": [10.0, 10.0],
            "volume": [1.0, 1.0],
        }
    )
    csv_path = tmp_path / "slope_volume.csv"
    price_data_frame.to_csv(csv_path, index=False)

    captured_arguments: dict[str, tuple[float, float] | None] = {
        "angle_range": None
    }

    def fake_attach_signals(
        frame: pandas.DataFrame,
        window_size: int = 40,
        angle_range: tuple[float, float] = (-16.69924423399362, 64.95379922035721),
    ) -> None:
        captured_arguments["angle_range"] = angle_range
        frame["ema_sma_cross_with_slope_and_volume_entry_signal"] = [True, False]
        frame["ema_sma_cross_with_slope_and_volume_exit_signal"] = [False, True]

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        trade = Trade(
            entry_date=date_index[0],
            exit_date=date_index[1],
            entry_price=10.0,
            exit_price=10.0,
            profit=0.0,
            holding_period=1,
        )
        return SimulationResult(trades=[trade], total_profit=0.0)

    monkeypatch.setattr(
        strategy,
        "attach_ema_sma_cross_with_slope_and_volume_signals",
        fake_attach_signals,
    )
    monkeypatch.setitem(
        strategy.BUY_STRATEGIES,
        "ema_sma_cross_with_slope_and_volume",
        fake_attach_signals,
    )
    monkeypatch.setitem(
        strategy.SELL_STRATEGIES,
        "ema_sma_cross_with_slope_and_volume",
        fake_attach_signals,
    )
    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)

    evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross_with_slope_and_volume_-26.6_26.6",
        "ema_sma_cross_with_slope_and_volume_-26.6_26.6",
    )

    assert captured_arguments["angle_range"] == (-26.6, 26.6)


def test_evaluate_combined_strategy_passes_near_and_above_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function should forward chip concentration ranges."""

    date_index = pandas.date_range("2020-01-01", periods=2, freq="D")
    price_data_frame = pandas.DataFrame(
        {"Date": date_index, "open": [10.0, 10.0], "close": [10.0, 10.0]}
    )
    csv_path = tmp_path / "chip.csv"
    price_data_frame.to_csv(csv_path, index=False)

    captured_parameters: dict[str, tuple[float, float] | None] = {
        "near_range": None,
        "above_range": None,
    }

    def fake_attach_signals(
        frame: pandas.DataFrame,
        window_size: int = 40,
        angle_range: tuple[float, float] = (
            -16.69924423399362,
            64.95379922035721,
        ),
        near_range: tuple[float, float] = (0.0, 0.12),
        above_range: tuple[float, float] = (0.0, 0.10),
    ) -> None:
        captured_parameters["near_range"] = near_range
        captured_parameters["above_range"] = above_range
        frame["ema_sma_cross_testing_entry_signal"] = [True, False]
        frame["ema_sma_cross_testing_exit_signal"] = [False, True]

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        trade = Trade(
            entry_date=date_index[0],
            exit_date=date_index[1],
            entry_price=10.0,
            exit_price=10.0,
            profit=0.0,
            holding_period=1,
        )
        return SimulationResult(trades=[trade], total_profit=0.0)

    monkeypatch.setattr(
        strategy, "attach_ema_sma_cross_testing_signals", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.BUY_STRATEGIES, "ema_sma_cross_testing", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.SELL_STRATEGIES, "ema_sma_cross_testing", fake_attach_signals
    )
    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)

    evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross_testing_40_-1_1_0.15,0.2_0.25,0.3",
        "ema_sma_cross_testing_40_-1_1_0.15,0.2_0.25,0.3",
    )

    assert captured_parameters["near_range"] == (0.15, 0.2)
    assert captured_parameters["above_range"] == (0.25, 0.3)


def test_evaluate_combined_strategy_renames_columns_with_angle_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Signal column names should include the slope range suffix."""

    date_index = pandas.date_range("2020-01-01", periods=2, freq="D")
    price_data_frame = pandas.DataFrame(
        {"Date": date_index, "open": [10.0, 10.0], "close": [10.0, 10.0]}
    )
    csv_path = tmp_path / "slope_rename.csv"
    price_data_frame.to_csv(csv_path, index=False)

    captured_column_names: list[str] = []

    def fake_attach_signals(
        frame: pandas.DataFrame,
        window_size: int = 40,
        angle_range: tuple[float, float] = (-16.69924423399362, 64.95379922035721),
    ) -> None:
        frame["ema_sma_cross_with_slope_entry_signal"] = [True, False]
        frame["ema_sma_cross_with_slope_exit_signal"] = [False, True]

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        passed_frame: pandas.DataFrame = kwargs["data"]
        captured_column_names.extend(passed_frame.columns)
        trade = Trade(
            entry_date=date_index[0],
            exit_date=date_index[1],
            entry_price=10.0,
            exit_price=10.0,
            profit=0.0,
            holding_period=1,
        )
        return SimulationResult(trades=[trade], total_profit=0.0)

    monkeypatch.setattr(
        strategy, "attach_ema_sma_cross_with_slope_signals", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.BUY_STRATEGIES, "ema_sma_cross_with_slope", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.SELL_STRATEGIES, "ema_sma_cross_with_slope", fake_attach_signals
    )
    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)

    evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross_with_slope_-26.6_26.6",
        "ema_sma_cross_with_slope_-26.6_26.6",
    )

    assert (
        "ema_sma_cross_with_slope_-26.6_26.6_entry_signal" in captured_column_names
    )
    assert (
        "ema_sma_cross_with_slope_-26.6_26.6_exit_signal" in captured_column_names
    )


def test_evaluate_combined_strategy_renames_columns_negative_positive_angle_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function should include negative-to-positive slope values in names."""

    date_index = pandas.date_range("2020-01-01", periods=2, freq="D")
    price_data_frame = pandas.DataFrame(
        {"Date": date_index, "open": [10.0, 10.0], "close": [10.0, 10.0]}
    )
    csv_path = tmp_path / "slope_negative_positive.csv"
    price_data_frame.to_csv(csv_path, index=False)

    captured_column_names: list[str] = []

    def fake_attach_signals(
        frame: pandas.DataFrame,
        window_size: int = 40,
        angle_range: tuple[float, float] = (-16.69924423399362, 64.95379922035721),
    ) -> None:
        frame["ema_sma_cross_with_slope_entry_signal"] = [True, False]
        frame["ema_sma_cross_with_slope_exit_signal"] = [False, True]

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        passed_frame: pandas.DataFrame = kwargs["data"]
        captured_column_names.extend(passed_frame.columns)
        trade = Trade(
            entry_date=date_index[0],
            exit_date=date_index[1],
            entry_price=10.0,
            exit_price=10.0,
            profit=0.0,
            holding_period=1,
        )
        return SimulationResult(trades=[trade], total_profit=0.0)

    monkeypatch.setattr(
        strategy, "attach_ema_sma_cross_with_slope_signals", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.BUY_STRATEGIES, "ema_sma_cross_with_slope", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.SELL_STRATEGIES, "ema_sma_cross_with_slope", fake_attach_signals
    )
    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)

    evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross_with_slope_-5.7_50.2",
        "ema_sma_cross_with_slope_-5.7_50.2",
    )

    assert (
        "ema_sma_cross_with_slope_-5.7_50.2_entry_signal" in captured_column_names
    )
    assert (
        "ema_sma_cross_with_slope_-5.7_50.2_exit_signal" in captured_column_names
    )

def test_evaluate_combined_strategy_dollar_volume_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function should skip symbols below the dollar volume threshold."""

    price_values = [10.0] * 60
    volume_values = [1_000_000] * 60
    date_index = pandas.date_range("2020-01-01", periods=60, freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": price_values,
            "close": price_values,
            "volume": volume_values,
        }
    )
    csv_path = tmp_path / "filtered.csv"
    price_data_frame.to_csv(csv_path, index=False)

    simulate_called: dict[str, bool] = {"called": False}

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        simulate_called["called"] = True
        return SimulationResult(trades=[], total_profit=0.0)

    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)

    result = evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross",
        "ema_sma_cross",
        minimum_average_dollar_volume=20,
    )
    assert result.total_trades == 0
    assert simulate_called["called"] is False

    simulate_called["called"] = False
    result = evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross",
        "ema_sma_cross",
        minimum_average_dollar_volume=5,
    )
    assert simulate_called["called"] is True


def test_evaluate_combined_strategy_dollar_volume_ratio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function should skip symbols below the dollar volume ratio."""

    volumes_by_symbol = {"AAA": 100_000_000, "BBB": 900_000_000}
    for symbol_name, volume_value in volumes_by_symbol.items():
        date_index = pandas.date_range("2020-01-01", periods=60, freq="D")
        pandas.DataFrame(
            {
                "Date": date_index,
                "open": [1.0] * 60,
                "close": [1.0] * 60,
                "volume": [volume_value] * 60,
                "symbol": [symbol_name] * 60,
            }
        ).to_csv(tmp_path / f"{symbol_name}.csv", index=False)

    processed_symbols: list[str] = []

    def fake_simulate_trades(
        data: pandas.DataFrame, *args: object, **kwargs: object
    ) -> SimulationResult:
        processed_symbols.append(data["symbol"].iloc[0])
        return SimulationResult(trades=[], total_profit=0.0)

    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)

    evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross",
        "ema_sma_cross",
        minimum_average_dollar_volume_ratio=0.2,
    )
    assert processed_symbols == ["BBB"]

    processed_symbols.clear()
    evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross",
        "ema_sma_cross",
        minimum_average_dollar_volume_ratio=0.05,
    )
    assert set(processed_symbols) == {"AAA", "BBB"}


def test_evaluate_combined_strategy_dollar_volume_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function should process only the top symbols by 50-day average dollar volume."""

    import stock_indicator.strategy as strategy_module

    for symbol_name, volume_value in {
        "AAA": 300_000_000,
        "BBB": 100_000_000,
        "CCC": 200_000_000,
    }.items():
        date_index = pandas.date_range("2020-01-01", periods=60, freq="D")
        pandas.DataFrame(
            {
                "Date": date_index,
                "open": [1.0] * 60,
                "close": [1.0] * 60,
                "volume": [volume_value] * 60,
                "symbol": [symbol_name] * 60,
            }
        ).to_csv(tmp_path / f"{symbol_name}.csv", index=False)

    processed_symbols: list[str] = []

    def fake_simulate_trades(
        data: pandas.DataFrame, *args: object, **kwargs: object
    ) -> SimulationResult:
        processed_symbols.append(data["symbol"].iloc[0])
        return SimulationResult(trades=[], total_profit=0.0)

    monkeypatch.setattr(strategy_module, "simulate_trades", fake_simulate_trades)
    monkeypatch.setattr(strategy_module, "BUY_STRATEGIES", {"noop": lambda df: None})
    monkeypatch.setattr(strategy_module, "SELL_STRATEGIES", {"noop": lambda df: None})
    monkeypatch.setattr(strategy_module, "SUPPORTED_STRATEGIES", {"noop": lambda df: None})

    strategy_module.evaluate_combined_strategy(
        tmp_path, "noop", "noop", top_dollar_volume_rank=2
    )
    assert set(processed_symbols) == {"AAA", "CCC"}


def test_build_eligibility_mask_respects_pick_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_build_eligibility_mask should honor the per-group pick limit."""

    import stock_indicator.strategy as strategy_module

    data_index = pandas.to_datetime(["2020-01-01"])
    volume_frame = pandas.DataFrame(
        {
            "AAA": [300.0],
            "BBB": [200.0],
            "CCC": [100.0],
            "DDD": [400.0],
            "EEE": [150.0],
            "FFF": [50.0],
        },
        index=data_index,
    )

    group_mapping = {
        "AAA": 1,
        "BBB": 1,
        "CCC": 1,
        "DDD": 2,
        "EEE": 2,
        "FFF": 3,
    }
    monkeypatch.setattr(
        strategy_module, "load_ff12_groups_by_symbol", lambda: group_mapping
    )

    mask = strategy_module._build_eligibility_mask(  # noqa: SLF001
        volume_frame,
        minimum_average_dollar_volume=None,
        top_dollar_volume_rank=4,
        minimum_average_dollar_volume_ratio=None,
        maximum_symbols_per_group=2,
    )
    selected_symbols = mask.columns[mask.iloc[0]].tolist()
    assert set(selected_symbols) == {"AAA", "BBB", "DDD", "EEE"}


def test_evaluate_combined_strategy_symbols_enter_and_exit_daily_universe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Symbols should appear only on days they rank within the daily universe."""

    import stock_indicator.strategy as strategy_module

    date_index = pandas.date_range("2020-01-01", periods=100, freq="D")
    volume_values_a = [200_000_000] * 50 + [50_000_000] * 50
    volume_values_b = [100_000_000] * 50 + [200_000_000] * 50
    for symbol_name, volume_values in {
        "AAA": volume_values_a,
        "BBB": volume_values_b,
    }.items():
        pandas.DataFrame(
            {
                "Date": date_index,
                "open": [1.0] * 100,
                "close": [1.0] * 100,
                "volume": volume_values,
                "symbol": [symbol_name] * 100,
            }
        ).to_csv(tmp_path / f"{symbol_name}.csv", index=False)

    captured_frames: dict[str, pandas.DataFrame] = {}

    def fake_simulate_trades(
        data: pandas.DataFrame, *args: object, **kwargs: object
    ) -> SimulationResult:
        captured_frames[data["symbol"].iloc[0]] = data
        return SimulationResult(trades=[], total_profit=0.0)

    monkeypatch.setattr(strategy_module, "simulate_trades", fake_simulate_trades)
    monkeypatch.setattr(strategy_module, "BUY_STRATEGIES", {"noop": lambda df: None})
    monkeypatch.setattr(strategy_module, "SELL_STRATEGIES", {"noop": lambda df: None})
    monkeypatch.setattr(strategy_module, "SUPPORTED_STRATEGIES", {"noop": lambda df: None})

    strategy_module.evaluate_combined_strategy(
        tmp_path, "noop", "noop", top_dollar_volume_rank=1
    )

    first_date = pandas.Timestamp("2020-03-09")
    later_date = pandas.Timestamp("2020-03-11")
    aaa_frame = captured_frames["AAA"]
    bbb_frame = captured_frames["BBB"]
    assert first_date in aaa_frame.index
    assert later_date in aaa_frame.index
    assert first_date in bbb_frame.index
    assert later_date in bbb_frame.index


def test_evaluate_combined_strategy_dollar_volume_filter_and_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only symbols above the dollar volume threshold should be ranked."""

    import stock_indicator.strategy as strategy_module

    volumes_by_symbol = {
        "AAA": 100_000_000,
        "BBB": 200_000_000,
        "CCC": 300_000_000,
        "DDD": 250_000_000,
    }
    for symbol_name, volume_value in volumes_by_symbol.items():
        date_index = pandas.date_range("2020-01-01", periods=60, freq="D")
        pandas.DataFrame(
            {
                "Date": date_index,
                "open": [1.0] * 60,
                "close": [1.0] * 60,
                "volume": [volume_value] * 60,
                "symbol": [symbol_name] * 60,
            }
        ).to_csv(tmp_path / f"{symbol_name}.csv", index=False)

    processed_symbols: list[str] = []
    captured_counts: list[int] = []

    def fake_simulate_trades(
        data: pandas.DataFrame, *args: object, **kwargs: object
    ) -> SimulationResult:
        processed_symbols.append(data["symbol"].iloc[0])
        return SimulationResult(trades=[], total_profit=0.0)

    def fake_simulate_portfolio_balance(
        trades: Iterable[Trade],
        starting_cash: float,
        maximum_position_count: int,
        withdraw_amount: float = 0.0,
    ) -> float:
        assert withdraw_amount == 0.0
        captured_counts.append(maximum_position_count)
        return starting_cash

    monkeypatch.setattr(strategy_module, "simulate_trades", fake_simulate_trades)
    monkeypatch.setattr(
        strategy_module, "simulate_portfolio_balance", fake_simulate_portfolio_balance
    )
    monkeypatch.setattr(strategy_module, "calculate_annual_returns", lambda *a, **k: {})
    monkeypatch.setattr(
        strategy_module, "calculate_annual_trade_counts", lambda *a, **k: {}
    )
    monkeypatch.setattr(strategy_module, "BUY_STRATEGIES", {"noop": lambda df: None})
    monkeypatch.setattr(strategy_module, "SELL_STRATEGIES", {"noop": lambda df: None})
    monkeypatch.setattr(strategy_module, "SUPPORTED_STRATEGIES", {"noop": lambda df: None})

    strategy_module.evaluate_combined_strategy(
        tmp_path,
        "noop",
        "noop",
        minimum_average_dollar_volume=150,
        top_dollar_volume_rank=2,
        maximum_position_count=2,
    )
    assert set(processed_symbols) == {"CCC", "DDD"}
    assert captured_counts == [2]


def test_evaluate_combined_strategy_filters_low_average_dollar_volume_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows below the dollar volume threshold should be excluded."""

    import stock_indicator.strategy as strategy_module

    date_index = pandas.date_range("2020-01-01", periods=52, freq="D")
    low_volume_date = date_index[50]
    open_values = [10.0] * 50 + [10.0, 30.0]
    close_values = [10.0] * 50 + [10.0, 30.0]
    volume_values = [10_000_000] * 50 + [0, 10_000_000]
    pandas.DataFrame(
        {
            "Date": date_index,
            "open": open_values,
            "close": close_values,
            "volume": volume_values,
        }
    ).to_csv(tmp_path / "AAA.csv", index=False)

    def fake_strategy(frame: pandas.DataFrame) -> None:
        frame["test_entry_signal"] = False
        frame["test_exit_signal"] = False
        # TODO: review

    captured_frames: dict[str, pandas.DataFrame] = {}

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        captured_frames["data"] = kwargs["data"]
        return SimulationResult(trades=[], total_profit=0.0)

    monkeypatch.setattr(strategy_module, "BUY_STRATEGIES", {"test": fake_strategy})
    monkeypatch.setattr(strategy_module, "SELL_STRATEGIES", {"test": fake_strategy})
    monkeypatch.setattr(strategy_module, "SUPPORTED_STRATEGIES", {"test": fake_strategy})
    monkeypatch.setattr(strategy_module, "simulate_trades", fake_simulate_trades)

    strategy_module.evaluate_combined_strategy(
        tmp_path,
        "test",
        "test",
        minimum_average_dollar_volume=100,
    )

    filtered_data = captured_frames["data"]
    assert low_volume_date in filtered_data.index
    assert (
        filtered_data.loc[low_volume_date, "simple_moving_average_dollar_volume"]
        / 1_000_000
    ) < 100


def test_evaluate_combined_strategy_trade_details_use_latest_average_dollar_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TradeDetail should store date-specific 50-day average dollar volumes."""

    import stock_indicator.strategy as strategy_module

    date_index = pandas.date_range("2020-01-01", periods=60, freq="D")
    pandas.DataFrame(
        {
            "Date": date_index,
            "open": [10.0] * 60,
            "close": [10.0] * 60,
            "volume": [1_000_000] * 59 + [2_000_000],
        }
    ).to_csv(tmp_path / "AAA.csv", index=False)
    pandas.DataFrame(
        {
            "Date": date_index,
            "open": [20.0] * 60,
            "close": [20.0] * 60,
            "volume": [2_000_000] * 59 + [4_000_000],
        }
    ).to_csv(tmp_path / "BBB.csv", index=False)

    monkeypatch.setattr(strategy_module, "BUY_STRATEGIES", {"noop": lambda frame: None})
    monkeypatch.setattr(strategy_module, "SELL_STRATEGIES", {"noop": lambda frame: None})
    monkeypatch.setattr(
        strategy_module, "SUPPORTED_STRATEGIES", {"noop": lambda frame: None}
    )

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        price_data_frame: pandas.DataFrame = kwargs["data"]
        entry_date = price_data_frame.index[58]
        exit_date = price_data_frame.index[59]
        entry_price = float(price_data_frame.iloc[58]["open"])
        exit_price = float(price_data_frame.iloc[59]["open"])
        trade = Trade(
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            profit=0.0,
            holding_period=1,
        )
        return SimulationResult(trades=[trade], total_profit=0.0)

    monkeypatch.setattr(strategy_module, "simulate_trades", fake_simulate_trades)

    result = strategy_module.evaluate_combined_strategy(tmp_path, "noop", "noop")
    trade_details = result.trade_details_by_year[2020]
    open_details = {
        detail.symbol: detail for detail in trade_details if detail.action == "open"
    }
    close_details = {
        detail.symbol: detail for detail in trade_details if detail.action == "close"
    }
    aaa_open = open_details["AAA"]
    bbb_open = open_details["BBB"]
    aaa_close = close_details["AAA"]
    bbb_close = close_details["BBB"]

    assert aaa_open.simple_moving_average_dollar_volume == pytest.approx(10_000_000.0)
    assert bbb_open.simple_moving_average_dollar_volume == pytest.approx(40_000_000.0)
    assert (
        aaa_open.total_simple_moving_average_dollar_volume
        == pytest.approx(50_000_000.0)
    )
    assert (
        bbb_open.total_simple_moving_average_dollar_volume
        == pytest.approx(50_000_000.0)
    )
    assert aaa_open.simple_moving_average_dollar_volume_ratio == pytest.approx(0.2)
    assert bbb_open.simple_moving_average_dollar_volume_ratio == pytest.approx(0.8)

    assert aaa_close.simple_moving_average_dollar_volume == pytest.approx(10_200_000.0)
    assert bbb_close.simple_moving_average_dollar_volume == pytest.approx(40_800_000.0)
    assert (
        aaa_close.total_simple_moving_average_dollar_volume
        == pytest.approx(51_000_000.0)
    )
    assert (
        bbb_close.total_simple_moving_average_dollar_volume
        == pytest.approx(51_000_000.0)
    )
    assert aaa_close.simple_moving_average_dollar_volume_ratio == pytest.approx(0.2)
    assert bbb_close.simple_moving_average_dollar_volume_ratio == pytest.approx(0.8)


def test_evaluate_combined_strategy_trade_details_use_signal_day_chip_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TradeDetail should log chip metrics from the prior trading day."""

    import stock_indicator.strategy as strategy_module

    date_index = pandas.date_range("2023-01-02", periods=61, freq="B")
    pandas.DataFrame(
        {
            "Date": date_index,
            "open": [10.0] * 61,
            "close": [10.0] * 61,
            "volume": [1_000_000] * 61,
        }
    ).to_csv(tmp_path / "AAA.csv", index=False)

    monkeypatch.setattr(strategy_module, "BUY_STRATEGIES", {"noop": lambda frame: None})
    monkeypatch.setattr(strategy_module, "SELL_STRATEGIES", {"noop": lambda frame: None})
    monkeypatch.setattr(
        strategy_module, "SUPPORTED_STRATEGIES", {"noop": lambda frame: None}
    )

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        price_data_frame: pandas.DataFrame = kwargs["data"]
        entry_date = price_data_frame.index[60]
        entry_price = float(price_data_frame.iloc[60]["open"])
        trade = Trade(
            entry_date=entry_date,
            exit_date=entry_date,
            entry_price=entry_price,
            exit_price=entry_price,
            profit=0.0,
            holding_period=0,
        )
        return SimulationResult(trades=[trade], total_profit=0.0)

    monkeypatch.setattr(strategy_module, "simulate_trades", fake_simulate_trades)

    signal_date = date_index[59]
    recorded_frame_end: pandas.Timestamp | None = None

    def fake_calculate_chip_concentration_metrics(
        frame: pandas.DataFrame,
        lookback_window_size: int = 60,
        bin_count: int = 50,
        near_price_band_ratio: float = 0.03,
        include_volume_profile: bool = False,
    ) -> dict[str, float | int | None]:
        nonlocal recorded_frame_end
        recorded_frame_end = frame.index[-1]
        return {
            "price_score": 1.0,
            "near_price_volume_ratio": 0.1,
            "above_price_volume_ratio": 0.2,
            "histogram_node_count": 5,
        }

    monkeypatch.setattr(
        strategy_module,
        "calculate_chip_concentration_metrics",
        fake_calculate_chip_concentration_metrics,
    )

    result = strategy_module.evaluate_combined_strategy(tmp_path, "noop", "noop")
    trade_details = result.trade_details_by_year[signal_date.year]
    open_detail = [
        detail for detail in trade_details if detail.action == "open"
    ][0]

    assert recorded_frame_end == signal_date
    assert open_detail.price_concentration_score == pytest.approx(1.0)
    assert open_detail.near_price_volume_ratio == pytest.approx(0.1)
    assert open_detail.above_price_volume_ratio == pytest.approx(0.2)
    assert open_detail.histogram_node_count == 5


def test_evaluate_combined_strategy_handles_empty_csv(tmp_path: Path) -> None:
    """evaluate_combined_strategy should skip empty CSV files and return zero trades."""
    empty_data_frame = pandas.DataFrame(
        columns=["Date", "open", "close", "volume"]
    )
    csv_file_path = tmp_path / "empty.csv"
    empty_data_frame.to_csv(csv_file_path, index=False)

    result = evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross",
        "ema_sma_cross",
    )

    assert result.total_trades == 0


def test_evaluate_combined_strategy_handles_blank_csv(tmp_path: Path) -> None:
    """evaluate_combined_strategy should skip CSV files without content."""
    csv_file_path = tmp_path / "blank.csv"
    csv_file_path.touch()

    result = evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross",
        "ema_sma_cross",
    )

    assert result.total_trades == 0


def test_evaluate_combined_strategy_reports_maximum_positions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function should return the highest number of overlapping trades."""
    import stock_indicator.strategy as strategy_module
    from stock_indicator.simulator import SimulationResult, Trade

    for symbol_name in ["AAA", "BBB"]:
        pandas.DataFrame(
            {
                "Date": ["2020-01-01"],
                "open": [1.0],
                "close": [1.0],
            }
        ).to_csv(tmp_path / f"{symbol_name}.csv", index=False)

    monkeypatch.setattr(strategy_module, "BUY_STRATEGIES", {"noop": lambda df: None})
    monkeypatch.setattr(strategy_module, "SELL_STRATEGIES", {"noop": lambda df: None})
    monkeypatch.setattr(strategy_module, "SUPPORTED_STRATEGIES", {"noop": lambda df: None})

    simulation_results = [
        SimulationResult(
            trades=[
                Trade(
                    entry_date=pandas.Timestamp("2020-01-01"),
                    exit_date=pandas.Timestamp("2020-01-03"),
                    entry_price=1.0,
                    exit_price=1.0,
                    profit=0.0,
                    holding_period=2,
                )
            ],
            total_profit=0.0,
        ),
        SimulationResult(
            trades=[
                Trade(
                    entry_date=pandas.Timestamp("2020-01-02"),
                    exit_date=pandas.Timestamp("2020-01-04"),
                    entry_price=1.0,
                    exit_price=1.0,
                    profit=0.0,
                    holding_period=2,
                )
            ],
            total_profit=0.0,
        ),
    ]
    simulation_iterator = iter(simulation_results)

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        return next(simulation_iterator)

    monkeypatch.setattr(strategy_module, "simulate_trades", fake_simulate_trades)

    result = strategy_module.evaluate_combined_strategy(tmp_path, "noop", "noop")
    assert result.maximum_concurrent_positions == 2


def test_evaluate_combined_strategy_skips_sp500_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """evaluate_combined_strategy should ignore the S&P 500 index symbol."""

    import stock_indicator.strategy as strategy_module
    from stock_indicator.symbols import SP500_SYMBOL

    date_index = pandas.date_range("2020-01-01", periods=60, freq="D")
    pandas.DataFrame(
        {
            "Date": date_index,
            "open": [1.0] * 60,
            "close": [1.0] * 60,
            "symbol": ["AAA"] * 60,
        }
    ).to_csv(tmp_path / "AAA.csv", index=False)
    pandas.DataFrame(
        {
            "Date": date_index,
            "open": [1.0] * 60,
            "close": [1.0] * 60,
            "symbol": [SP500_SYMBOL] * 60,
        }
    ).to_csv(tmp_path / f"{SP500_SYMBOL}.csv", index=False)

    processed_symbol_names: list[str] = []

    def fake_simulate_trades(
        data: pandas.DataFrame, *args: object, **kwargs: object
    ) -> SimulationResult:
        processed_symbol_names.append(data["symbol"].iloc[0])
        return SimulationResult(trades=[], total_profit=0.0)

    monkeypatch.setattr(strategy_module, "simulate_trades", fake_simulate_trades)
    monkeypatch.setattr(strategy_module, "BUY_STRATEGIES", {"noop": lambda frame: None})
    monkeypatch.setattr(strategy_module, "SELL_STRATEGIES", {"noop": lambda frame: None})
    monkeypatch.setattr(strategy_module, "SUPPORTED_STRATEGIES", {"noop": lambda frame: None})

    strategy_module.evaluate_combined_strategy(tmp_path, "noop", "noop")
    assert processed_symbol_names == ["AAA"]


def test_evaluate_combined_strategy_ignores_trades_before_start_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """evaluate_combined_strategy should ignore trades before the start date."""

    import stock_indicator.strategy as strategy_module

    date_index = pandas.date_range("2020-01-01", periods=5, freq="D")
    pandas.DataFrame(
        {
            "Date": date_index,
            "open": [10.0, 10.0, 10.0, 10.0, 12.0],
            "close": [10.0, 10.0, 10.0, 10.0, 12.0],
            "volume": [1_000_000] * 5,
        }
    ).to_csv(tmp_path / "AAA.csv", index=False)

    def fake_strategy(frame: pandas.DataFrame) -> None:
        frame["test_entry_signal"] = False
        frame["test_exit_signal"] = False
        if pandas.Timestamp("2020-01-01") in frame.index:
            frame.loc[pandas.Timestamp("2020-01-01"), "test_entry_signal"] = True
        if pandas.Timestamp("2020-01-02") in frame.index:
            frame.loc[pandas.Timestamp("2020-01-02"), "test_exit_signal"] = True
        if pandas.Timestamp("2020-01-04") in frame.index:
            frame.loc[pandas.Timestamp("2020-01-04"), "test_entry_signal"] = True
        if pandas.Timestamp("2020-01-05") in frame.index:
            frame.loc[pandas.Timestamp("2020-01-05"), "test_exit_signal"] = True

    monkeypatch.setattr(strategy_module, "BUY_STRATEGIES", {"test": fake_strategy})
    monkeypatch.setattr(strategy_module, "SELL_STRATEGIES", {"test": fake_strategy})
    monkeypatch.setattr(strategy_module, "SUPPORTED_STRATEGIES", {"test": fake_strategy})

    result_without_filter = strategy_module.evaluate_combined_strategy(
        tmp_path, "test", "test"
    )
    result_with_filter = strategy_module.evaluate_combined_strategy(
        tmp_path,
        "test",
        "test",
        start_date=pandas.Timestamp("2020-01-03"),
    )

    assert result_without_filter.total_trades == 2
    assert result_with_filter.total_trades == 1
    for trade_details in result_with_filter.trade_details_by_year.values():
        for detail in trade_details:
            assert detail.date >= pandas.Timestamp("2020-01-03")


def test_attach_20_50_sma_cross_signals_mark_crosses() -> None:
    """The 20/50 SMA cross signals should mark upward and downward crosses."""
    # TODO: review

    import stock_indicator.strategy as strategy_module
    from stock_indicator.indicators import sma

    close_values = (
        [100.0] * 50
        + list(range(100, 50, -1))
        + list(range(50, 200))
        + list(range(200, 50, -1))
    )
    price_data_frame = pandas.DataFrame({"close": close_values})

    strategy_module.attach_20_50_sma_cross_signals(price_data_frame)

    sma_20_series = sma(price_data_frame["close"], 20)
    sma_50_series = sma(price_data_frame["close"], 50)
    expected_entry_series = (
        (sma_20_series.shift(1) <= sma_50_series.shift(1))
        & (sma_20_series > sma_50_series)
    ).shift(1, fill_value=False)
    expected_exit_series = (
        (sma_20_series.shift(1) >= sma_50_series.shift(1))
        & (sma_20_series < sma_50_series)
    ).shift(1, fill_value=False)

    assert price_data_frame["20_50_sma_cross_entry_signal"].equals(expected_entry_series)
    assert price_data_frame["20_50_sma_cross_exit_signal"].equals(expected_exit_series)
    assert price_data_frame["20_50_sma_cross_entry_signal"].any()
    assert price_data_frame["20_50_sma_cross_exit_signal"].any()


def test_attach_parametrized_20_50_sma_cross_uses_custom_windows() -> None:
    """Custom short/long windows should override 20/50 defaults."""
    # TODO: review

    import stock_indicator.strategy as strategy_module
    from stock_indicator.indicators import sma

    close_values = (
        [100.0] * 50
        + list(range(100, 50, -1))
        + list(range(50, 200))
        + list(range(200, 50, -1))
    )
    price_data_frame = pandas.DataFrame({"close": close_values})

    # Use 15/30 instead of 20/50
    strategy_module.attach_20_50_sma_cross_signals(
        price_data_frame, short_window_size=15, long_window_size=30
    )

    sma_short = sma(price_data_frame["close"], 15)
    sma_long = sma(price_data_frame["close"], 30)
    expected_entry = (
        (sma_short.shift(1) <= sma_long.shift(1)) & (sma_short > sma_long)
    ).shift(1, fill_value=False)
    expected_exit = (
        (sma_short.shift(1) >= sma_long.shift(1)) & (sma_short < sma_long)
    ).shift(1, fill_value=False)

    assert price_data_frame["20_50_sma_cross_entry_signal"].equals(expected_entry)
    assert price_data_frame["20_50_sma_cross_exit_signal"].equals(expected_exit)


def test_attach_ema_sma_cross_and_rsi_signals_filters_by_rsi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The EMA/SMA cross entry signal should require RSI to be 40 or below."""
    # TODO: review

    import stock_indicator.strategy as strategy_module

    price_data_frame = pandas.DataFrame(
        {"open": [1.0, 1.0, 1.0], "close": [1.0, 1.0, 1.0]}
    )

    def fake_attach_ema_sma_cross_signals(
        data_frame: pandas.DataFrame,
        window_size: int = 50,
        require_close_above_long_term_sma: bool = True,
    ) -> None:
        data_frame["ema_sma_cross_entry_signal"] = pandas.Series(
            [False, True, True]
        )
        data_frame["ema_sma_cross_exit_signal"] = pandas.Series(
            [False, False, True]
        )

    def fake_rsi(
        price_series: pandas.Series, window_size: int = 14
    ) -> pandas.Series:
        return pandas.Series([50, 30, 50])

    monkeypatch.setattr(
        strategy_module, "attach_ema_sma_cross_signals", fake_attach_ema_sma_cross_signals
    )
    monkeypatch.setattr(strategy_module, "rsi", fake_rsi)

    strategy_module.attach_ema_sma_cross_and_rsi_signals(price_data_frame)

    assert list(price_data_frame["ema_sma_cross_and_rsi_entry_signal"]) == [
        False,
        True,
        False,
    ]
    assert list(price_data_frame["ema_sma_cross_and_rsi_exit_signal"]) == [
        False,
        False,
        True,
    ]


def test_attach_ftd_ema_sma_cross_signals_requires_recent_ftd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The FTD/EMA-SMA cross entry signal should require a recent FTD."""
    # TODO: review

    import stock_indicator.strategy as strategy_module

    price_data_frame = pandas.DataFrame(
        {"open": [1.0] * 7, "close": [1.0] * 7}
    )

    def fake_attach_ema_sma_cross_signals(
        data_frame: pandas.DataFrame,
        window_size: int = 50,
        require_close_above_long_term_sma: bool = True,
    ) -> None:
        data_frame["ema_sma_cross_entry_signal"] = pandas.Series(
            [False, False, False, False, True, False, True]
        )
        data_frame["ema_sma_cross_exit_signal"] = pandas.Series(
            [False, False, False, False, False, False, False]
        )

    def fake_ftd(
        data_frame: pandas.DataFrame, buy_mark_day: int, tolerance: float = 1e-8
    ) -> bool:
        return len(data_frame) - 1 == 1

    monkeypatch.setattr(
        strategy_module, "attach_ema_sma_cross_signals", fake_attach_ema_sma_cross_signals
    )
    monkeypatch.setattr(strategy_module, "ftd", fake_ftd)

    strategy_module.attach_ftd_ema_sma_cross_signals(price_data_frame)

    assert list(price_data_frame["ftd_ema_sma_cross_entry_signal"]) == [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert list(price_data_frame["ftd_ema_sma_cross_exit_signal"]) == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_attach_ema_sma_cross_with_slope_filters_by_angle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The EMA/SMA cross entry applies angle filters and requires price above the SMA."""
    # TODO: review

    import stock_indicator.strategy as strategy_module

    price_data_frame = pandas.DataFrame(
        {
            "open": [1.0, 1.0, 1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0, 1.0, 1.0],
            "volume": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    recorded_require_close_above_long_term_sma: bool | None = None

    def fake_attach_ema_sma_cross_signals(
        data_frame: pandas.DataFrame,
        window_size: int = 50,
        require_close_above_long_term_sma: bool = True,
        sma_window_factor: float | None = None,
    ) -> None:
        nonlocal recorded_require_close_above_long_term_sma
        recorded_require_close_above_long_term_sma = require_close_above_long_term_sma
        data_frame["sma_value"] = pandas.Series(
            [1.0, 0.8, 0.9, 1.2, 0.7]
        )
        data_frame["sma_previous"] = data_frame["sma_value"].shift(1)
        data_frame["ema_sma_cross_entry_signal"] = pandas.Series(
            [False, True, True, True, True]
        )
        data_frame["ema_sma_cross_exit_signal"] = pandas.Series(
            [False, False, False, False, True]
        )

    monkeypatch.setattr(
        strategy_module, "attach_ema_sma_cross_signals", fake_attach_ema_sma_cross_signals
    )

    strategy_module.attach_ema_sma_cross_with_slope_signals(
        price_data_frame, angle_range=(0.0, 0.2), bounds_as_tangent=True
    )

    assert recorded_require_close_above_long_term_sma is True
    assert list(price_data_frame["ema_sma_cross_with_slope_entry_signal"]) == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert list(price_data_frame["ema_sma_cross_with_slope_exit_signal"]) == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_attach_ema_sma_cross_with_slope_signals_raises_value_error_for_invalid_angle_range() -> None:
    """``attach_ema_sma_cross_with_slope_signals`` should validate the angle range."""
    # TODO: review

    import stock_indicator.strategy as strategy_module

    price_data_frame = pandas.DataFrame({"open": [1.0], "close": [1.0]})

    with pytest.raises(
        ValueError, match="lower bound cannot exceed upper bound"
    ):
        strategy_module.attach_ema_sma_cross_with_slope_signals(
            price_data_frame, angle_range=(1.0, -1.0)
        )


def test_attach_ema_sma_cross_testing_filters_by_angle_and_chip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entry applies angle and chip concentration filters without long-term SMA."""
    # TODO: review

    import stock_indicator.strategy as strategy_module

    price_data_frame = pandas.DataFrame(
        {
            "open": [1.0, 1.0, 1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0, 1.0, 1.0],
            "volume": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    recorded_flag: bool | None = None

    def fake_attach_ema_sma_cross_signals(
        data_frame: pandas.DataFrame,
        window_size: int = 50,
        require_close_above_long_term_sma: bool = True,
        sma_window_factor: float | None = None,
        include_raw_signals: bool = False,
    ) -> None:
        nonlocal recorded_flag
        recorded_flag = require_close_above_long_term_sma
        data_frame["sma_value"] = pandas.Series([1.0, 0.8, 0.9, 1.3, 1.2])
        data_frame["sma_previous"] = data_frame["sma_value"].shift(1)
        data_frame["ema_sma_cross_entry_signal"] = pandas.Series(
            [False, True, True, True, True]
        )
        data_frame["ema_sma_cross_exit_signal"] = pandas.Series(
            [False, False, False, False, True]
        )

    metrics_queue = [
        {"near_price_volume_ratio": 0.2, "above_price_volume_ratio": 0.2},
        {"near_price_volume_ratio": 0.11, "above_price_volume_ratio": 0.09},
        {"near_price_volume_ratio": 0.05, "above_price_volume_ratio": 0.11},
        {"near_price_volume_ratio": 0.13, "above_price_volume_ratio": 0.09},
        {"near_price_volume_ratio": 0.2, "above_price_volume_ratio": 0.2},
    ]

    def fake_calculate_chip_concentration_metrics(
        frame: pandas.DataFrame,
        lookback_window_size: int = 60,
        bin_count: int = 50,
        near_price_band_ratio: float = 0.03,
        include_volume_profile: bool = False,
    ) -> dict[str, float | int | None]:
        return metrics_queue.pop(0)

    monkeypatch.setattr(
        strategy_module, "attach_ema_sma_cross_signals", fake_attach_ema_sma_cross_signals
    )
    monkeypatch.setattr(
        strategy_module,
        "calculate_chip_concentration_metrics",
        fake_calculate_chip_concentration_metrics,
    )

    strategy_module.attach_ema_sma_cross_testing_signals(
        price_data_frame, angle_range=(0.0, 0.2), bounds_as_tangent=True
    )

    assert recorded_flag is False
    assert list(price_data_frame["ema_sma_cross_testing_entry_signal"]) == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert list(price_data_frame["ema_sma_cross_testing_exit_signal"]) == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_attach_ema_sma_cross_testing_signals_raises_value_error_for_invalid_angle_range() -> None:
    """``attach_ema_sma_cross_testing_signals`` should validate the slope range."""
    # TODO: review

    import stock_indicator.strategy as strategy_module

    price_data_frame = pandas.DataFrame({"open": [1.0], "close": [1.0]})

    with pytest.raises(
        ValueError, match="lower bound cannot exceed upper bound",
    ):
        strategy_module.attach_ema_sma_cross_testing_signals(
            price_data_frame, angle_range=(1.0, -1.0)
        )


def test_attach_ema_sma_cross_testing_uses_previous_day_ratios_on_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entry uses prior-day chip ratios when a weekend gap separates cross and entry."""
    import stock_indicator.strategy as strategy_module

    trading_dates = pandas.to_datetime(
        ["2023-05-18", "2023-05-19", "2023-05-22"]
    )
    price_data_frame = pandas.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1.0, 1.0, 1.0],
        },
        index=trading_dates,
    )

    def fake_attach_ema_sma_cross_signals(
        data_frame: pandas.DataFrame,
        window_size: int = 40,
        require_close_above_long_term_sma: bool = False,
        sma_window_factor: float | None = None,
        include_raw_signals: bool = False,
    ) -> None:
        data_frame["sma_value"] = pandas.Series(
            [1.0, 1.1, 1.2], index=data_frame.index
        )
        data_frame["sma_previous"] = data_frame["sma_value"].shift(1)
        data_frame["ema_sma_cross_entry_signal"] = pandas.Series(
            [False, False, True], index=data_frame.index
        )
        data_frame["ema_sma_cross_exit_signal"] = pandas.Series(
            [False, False, False], index=data_frame.index
        )
        data_frame["ema_sma_cross_raw_entry_signal"] = pandas.Series(
            [False, True, False], index=data_frame.index
        )
        data_frame["ema_sma_cross_raw_exit_signal"] = pandas.Series(
            [False, False, False], index=data_frame.index
        )

    metrics_queue = [
        {"near_price_volume_ratio": 0.2, "above_price_volume_ratio": 0.2},
        {"near_price_volume_ratio": 0.11, "above_price_volume_ratio": 0.09},
        {"near_price_volume_ratio": 0.5, "above_price_volume_ratio": 0.5},
    ]

    def fake_calculate_chip_concentration_metrics(
        frame: pandas.DataFrame,
        lookback_window_size: int = 60,
        bin_count: int = 50,
        near_price_band_ratio: float = 0.03,
        include_volume_profile: bool = False,
    ) -> dict[str, float | int | None]:
        return metrics_queue.pop(0)

    monkeypatch.setattr(
        strategy_module, "attach_ema_sma_cross_signals", fake_attach_ema_sma_cross_signals
    )
    monkeypatch.setattr(
        strategy_module,
        "calculate_chip_concentration_metrics",
        fake_calculate_chip_concentration_metrics,
    )

    strategy_module.attach_ema_sma_cross_testing_signals(
        price_data_frame, include_raw_signals=True
    )

    assert price_data_frame.loc[trading_dates[2], "ema_sma_cross_testing_entry_signal"]
    assert not price_data_frame.loc[
        trading_dates[2], "ema_sma_cross_testing_raw_entry_signal"
    ]


def test_attach_ema_sma_cross_testing_respects_range_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Signals trigger when either chip ratio falls within bounds."""

    import stock_indicator.strategy as strategy_module

    price_data_frame = pandas.DataFrame(
        {
            "open": [1.0, 1.0, 1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0, 1.0, 1.0],
            "volume": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    def fake_attach_cross_signals(
        data_frame: pandas.DataFrame,
        window_size: int = 40,
        require_close_above_long_term_sma: bool = False,
        sma_window_factor: float | None = None,
        include_raw_signals: bool = False,
    ) -> None:
        data_frame["sma_value"] = pandas.Series([1.0] * 5)
        data_frame["sma_previous"] = data_frame["sma_value"].shift(1)
        data_frame["ema_sma_cross_entry_signal"] = pandas.Series(
            [False, True, True, True, True]
        )
        data_frame["ema_sma_cross_exit_signal"] = pandas.Series(
            [False, False, False, False, False]
        )

    metrics_queue = [
        {"near_price_volume_ratio": 0.2, "above_price_volume_ratio": 0.2},
        {"near_price_volume_ratio": 0.04, "above_price_volume_ratio": 0.06},
        {"near_price_volume_ratio": 0.06, "above_price_volume_ratio": 0.04},
        {"near_price_volume_ratio": 0.07, "above_price_volume_ratio": 0.07},
        {"near_price_volume_ratio": 0.07, "above_price_volume_ratio": 0.07},
    ]

    def fake_metrics(*_: object, **__: object) -> dict[str, float | int | None]:
        return metrics_queue.pop(0)

    monkeypatch.setattr(
        strategy_module, "attach_ema_sma_cross_signals", fake_attach_cross_signals
    )
    monkeypatch.setattr(
        strategy_module,
        "calculate_chip_concentration_metrics",
        fake_metrics,
    )

    strategy_module.attach_ema_sma_cross_testing_signals(
        price_data_frame,
        near_range=(0.05, 0.08),
        above_range=(0.05, 0.08),
    )

    assert list(price_data_frame["ema_sma_cross_testing_entry_signal"]) == [
        False,
        False,
        True,
        True,
        True,
    ]


def test_generate_strategy_artifacts_use_run_frame_index_for_signal_date(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Trade details should record chip ratios from the prior run-frame bar."""

    date_index = pandas.to_datetime(
        [
            "2025-09-01",
            "2025-09-02",
            "2025-09-03",
            "2025-09-04",
            "2025-09-05",
            "2025-09-08",
        ]
    )
    price_rows = pandas.DataFrame(
        {
            "Date": date_index,
            "Open": [100.0, 100.5, 101.0, 101.5, 102.0, 102.5],
            "High": [101.0, 101.5, 102.0, 102.5, 103.0, 103.5],
            "Low": [99.5, 100.0, 100.5, 101.0, 101.5, 102.0],
            "Close": [100.5, 101.0, 101.5, 102.0, 102.5, 103.0],
            "Volume": [1_000_000, 1_050_000, 1_100_000, 1_150_000, 1_200_000, 1_250_000],
        }
    )
    csv_path = tmp_path / "TEST.csv"
    price_rows.to_csv(csv_path, index=False)

    metrics_by_date = {
        pandas.Timestamp("2025-09-01"): {
            "price_score": 0.41,
            "near_price_volume_ratio": 0.91,
            "above_price_volume_ratio": 0.88,
            "histogram_node_count": 5,
        },
        pandas.Timestamp("2025-09-02"): {
            "price_score": 0.42,
            "near_price_volume_ratio": 0.82,
            "above_price_volume_ratio": 0.79,
            "histogram_node_count": 5,
        },
        pandas.Timestamp("2025-09-03"): {
            "price_score": 0.43,
            "near_price_volume_ratio": 0.73,
            "above_price_volume_ratio": 0.7,
            "histogram_node_count": 5,
        },
        pandas.Timestamp("2025-09-04"): {
            "price_score": 0.31,
            "near_price_volume_ratio": 0.25,
            "above_price_volume_ratio": 0.35,
            "histogram_node_count": 6,
        },
        pandas.Timestamp("2025-09-05"): {
            "price_score": 0.29,
            "near_price_volume_ratio": 0.92,
            "above_price_volume_ratio": 0.87,
            "histogram_node_count": 6,
        },
        pandas.Timestamp("2025-09-08"): {
            "price_score": 0.28,
            "near_price_volume_ratio": 0.6,
            "above_price_volume_ratio": 0.58,
            "histogram_node_count": 6,
        },
    }

    def fake_chip_metrics(
        frame: pandas.DataFrame,
        lookback_window_size: int = 60,
        bin_count: int = 50,
        near_price_band_ratio: float = 0.03,
        include_volume_profile: bool = False,
    ) -> dict[str, float | int | None]:
        last_bar_date = frame.index[-1]
        metrics = metrics_by_date[last_bar_date]
        return {
            "price_score": metrics["price_score"],
            "near_price_volume_ratio": metrics["near_price_volume_ratio"],
            "above_price_volume_ratio": metrics["above_price_volume_ratio"],
            "histogram_node_count": metrics["histogram_node_count"],
        }

    def fake_simulate_trades(*_: object, **__: object) -> SimulationResult:
        trade_entry_date = pandas.Timestamp("2025-09-05")
        trade_exit_date = pandas.Timestamp("2025-09-08")
        trade = Trade(
            entry_date=trade_entry_date,
            exit_date=trade_exit_date,
            entry_price=102.5,
            exit_price=103.5,
            profit=1.0,
            holding_period=(trade_exit_date - trade_entry_date).days,
            exit_reason="signal",
        )
        return SimulationResult(trades=[trade], total_profit=1.0)

    def simple_sma(series: pandas.Series, window_size: int) -> pandas.Series:
        return series.rolling(window=window_size, min_periods=1).mean()

    monkeypatch.setattr(strategy, "calculate_chip_concentration_metrics", fake_chip_metrics)
    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)
    monkeypatch.setattr(strategy, "load_symbols_excluded_by_industry", lambda: set())
    monkeypatch.setattr(strategy, "load_ff12_groups_by_symbol", lambda: {})
    monkeypatch.setattr(strategy, "sma", simple_sma)

    artifacts = strategy._generate_strategy_evaluation_artifacts(
        tmp_path,
        "ema_sma_cross_testing",
        "ema_sma_cross_testing",
        start_date=pandas.Timestamp("2025-09-04"),
    )

    assert artifacts.trade_detail_pairs
    entry_detail, _ = next(iter(artifacts.trade_detail_pairs.values()))
    expected_metrics = metrics_by_date[pandas.Timestamp("2025-09-04")]

    assert entry_detail.near_price_volume_ratio == pytest.approx(
        expected_metrics["near_price_volume_ratio"]
    )
    assert entry_detail.above_price_volume_ratio == pytest.approx(
        expected_metrics["above_price_volume_ratio"]
    )

def test_attach_ema_sma_cross_with_slope_and_volume_requires_higher_ema_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entry requires EMA dollar volume to exceed the SMA value."""
    # TODO: review

    import stock_indicator.strategy as strategy_module

    price_data_frame = pandas.DataFrame(
        {
            "open": [1.0, 1.0, 1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0, 1.0, 1.0],
            "volume": [1.0, 2.0, 3.0, 2.0, 1.0],
        }
    )

    def fake_attach_ema_sma_cross_with_slope_signals(
        data_frame: pandas.DataFrame,
        window_size: int = 50,
        angle_range: tuple[float, float] = (-16.69924423399362, 64.95379922035721),
    ) -> None:
        data_frame["ema_sma_cross_with_slope_entry_signal"] = pandas.Series(
            [False, True, True, True, True]
        )
        data_frame["ema_sma_cross_with_slope_exit_signal"] = pandas.Series(
            [False, False, False, False, True]
        )

    monkeypatch.setattr(
        strategy_module,
        "attach_ema_sma_cross_with_slope_signals",
        fake_attach_ema_sma_cross_with_slope_signals,
    )

    strategy_module.attach_ema_sma_cross_with_slope_and_volume_signals(
        price_data_frame, window_size=3
    )

    assert list(
        price_data_frame["ema_sma_cross_with_slope_and_volume_entry_signal"]
    ) == [False, False, True, False, False]
    assert list(
        price_data_frame["ema_sma_cross_with_slope_and_volume_exit_signal"]
    ) == [False, False, False, False, True]


def test_attach_ema_sma_cross_with_slope_and_volume_signals_raises_value_error_for_invalid_angle_range() -> None:
    """``attach_ema_sma_cross_with_slope_and_volume_signals`` should validate the slope range."""
    # TODO: review

    import stock_indicator.strategy as strategy_module

    price_data_frame = pandas.DataFrame(
        {"open": [1.0], "close": [1.0], "volume": [1.0]}
    )

    with pytest.raises(
        ValueError, match="lower bound cannot exceed upper bound"
    ):
        strategy_module.attach_ema_sma_cross_with_slope_and_volume_signals(
            price_data_frame, angle_range=(1.0, -1.0)
        )


def test_attach_ema_sma_double_cross_requires_long_term_ema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The double cross entry should require the long-term EMA above the SMA."""
    # TODO: review

    import stock_indicator.strategy as strategy_module

    price_data_frame = pandas.DataFrame(
        {"open": [1.0, 1.0, 1.0], "close": [1.0, 1.0, 1.0]}
    )

    recorded_require_close: bool | None = None

    def fake_attach_ema_sma_cross_signals(
        data_frame: pandas.DataFrame,
        window_size: int = 50,
        require_close_above_long_term_sma: bool = True,
    ) -> None:
        nonlocal recorded_require_close
        recorded_require_close = require_close_above_long_term_sma
        data_frame["ema_sma_cross_entry_signal"] = pandas.Series(
            [False, True, True]
        )
        data_frame["ema_sma_cross_exit_signal"] = pandas.Series(
            [False, False, True]
        )
        data_frame["long_term_sma_previous"] = pandas.Series([1.0, 1.0, 1.0])

    def fake_ema(
        price_series: pandas.Series, window_size: int
    ) -> pandas.Series:
        if window_size == strategy_module.LONG_TERM_SMA_WINDOW:
            return pandas.Series([1.0, 2.0, 0.5])
        return pandas.Series([0.0, 0.0, 0.0])

    monkeypatch.setattr(
        strategy_module, "attach_ema_sma_cross_signals", fake_attach_ema_sma_cross_signals
    )
    monkeypatch.setattr(strategy_module, "ema", fake_ema)

    strategy_module.attach_ema_sma_double_cross_signals(price_data_frame)

    assert recorded_require_close is False
    assert list(price_data_frame["ema_sma_double_cross_entry_signal"]) == [
        False,
        False,
        True,
    ]
    assert list(price_data_frame["ema_sma_double_cross_exit_signal"]) == [
        False,
        False,
        True,
    ]


def test_supported_strategies_includes_20_50_sma_cross() -> None:
    """``SUPPORTED_STRATEGIES`` should expose the 20/50 SMA cross strategy."""
    # TODO: review

    from stock_indicator.strategy import (
        SUPPORTED_STRATEGIES,
        attach_20_50_sma_cross_signals,
    )

    assert (
        SUPPORTED_STRATEGIES["20_50_sma_cross"]
        is attach_20_50_sma_cross_signals
    )


def test_supported_strategies_includes_ftd_ema_sma_cross() -> None:
    """``SUPPORTED_STRATEGIES`` should expose the FTD/EMA-SMA cross strategy."""
    # TODO: review

    from stock_indicator.strategy import (
        SUPPORTED_STRATEGIES,
        attach_ftd_ema_sma_cross_signals,
    )

    assert (
        SUPPORTED_STRATEGIES["ftd_ema_sma_cross"]
        is attach_ftd_ema_sma_cross_signals
    )


def test_supported_strategies_includes_ema_sma_cross_with_slope() -> None:
    """``SUPPORTED_STRATEGIES`` should expose the EMA/SMA cross with slope strategy."""
    # TODO: review

    from stock_indicator.strategy import (
        SUPPORTED_STRATEGIES,
        attach_ema_sma_cross_with_slope_signals,
    )

    assert (
        SUPPORTED_STRATEGIES["ema_sma_cross_with_slope"]
        is attach_ema_sma_cross_with_slope_signals
    )


def test_supported_strategies_includes_ema_sma_cross_testing() -> None:
    """``SUPPORTED_STRATEGIES`` should expose the testing strategy."""
    # TODO: review

    from stock_indicator.strategy import (
        SUPPORTED_STRATEGIES,
        attach_ema_sma_cross_testing_signals,
    )

    assert (
        SUPPORTED_STRATEGIES["ema_sma_cross_testing"]
        is attach_ema_sma_cross_testing_signals
    )


def test_supported_strategies_includes_ema_sma_cross_with_slope_and_volume() -> None:
    """``SUPPORTED_STRATEGIES`` should expose the slope and volume strategy."""
    # TODO: review

    from stock_indicator.strategy import (
        SUPPORTED_STRATEGIES,
        attach_ema_sma_cross_with_slope_and_volume_signals,
    )

    assert (
        SUPPORTED_STRATEGIES["ema_sma_cross_with_slope_and_volume"]
        is attach_ema_sma_cross_with_slope_and_volume_signals
    )


def test_supported_strategies_includes_ema_sma_double_cross() -> None:
    """``SUPPORTED_STRATEGIES`` should expose the EMA/SMA double cross strategy."""
    # TODO: review

    from stock_indicator.strategy import (
        SUPPORTED_STRATEGIES,
        attach_ema_sma_double_cross_signals,
    )

    assert (
        SUPPORTED_STRATEGIES["ema_sma_double_cross"]
        is attach_ema_sma_double_cross_signals
    )


def test_parse_strategy_name_with_window_size() -> None:
    """``parse_strategy_name`` should parse the window size suffix."""

    base_name, window_size, angle_range, near_range, above_range = parse_strategy_name(
        "ema_sma_cross_with_slope_40"
    )
    assert base_name == "ema_sma_cross_with_slope"
    assert window_size == 40
    assert angle_range is None
    assert near_range is None
    assert above_range is None


def test_parse_strategy_name_with_window_and_angle_range() -> None:
    """The parser should extract both window size and angle range."""

    base_name, window_size, angle_range, near_range, above_range = parse_strategy_name(
        "ema_sma_cross_with_slope_40_-26.6_26.6"
    )
    assert base_name == "ema_sma_cross_with_slope"
    assert window_size == 40
    assert angle_range == (-26.6, 26.6)
    assert near_range is None
    assert above_range is None


def test_parse_strategy_name_with_angle_range_only() -> None:
    """The parser should handle angle range without window size."""

    base_name, window_size, angle_range, near_range, above_range = parse_strategy_name(
        "ema_sma_cross_with_slope_-26.6_26.6"
    )
    assert base_name == "ema_sma_cross_with_slope"
    assert window_size is None
    assert angle_range == (-26.6, 26.6)
    assert near_range is None
    assert above_range is None


def test_parse_strategy_name_with_integer_slope_values() -> None:
    """``parse_strategy_name`` should convert integer slope bounds to floats."""

    base_name, window_size, angle_range, near_range, above_range = parse_strategy_name(
        "ema_sma_cross_with_slope_-1_2"
    )
    assert base_name == "ema_sma_cross_with_slope"
    assert window_size is None
    assert angle_range == (-1.0, 2.0)
    assert near_range is None
    assert above_range is None


def test_parse_strategy_name_with_all_segments() -> None:
    """The parser should handle window, angle range, and percentage thresholds."""

    (
        base_name,
        window_size,
        angle_range,
        near_range,
        above_range,
    ) = parse_strategy_name("ema_sma_cross_with_slope_40_-26.6_26.6_0.5_1.0")
    assert base_name == "ema_sma_cross_with_slope"
    assert window_size == 40
    assert angle_range == (-26.6, 26.6)
    assert near_range == (0.0, 0.5)
    assert above_range == (0.0, 1.0)


def test_parse_strategy_name_with_near_and_above_thresholds() -> None:
    """The parser should extract near and above range bounds."""

    (
        base_name,
        window_size,
        angle_range,
        near_range,
        above_range,
    ) = parse_strategy_name(
        "ema_sma_cross_testing_40_-26.6_26.6_0.11,0.12_0.09,0.1"
    )
    assert base_name == "ema_sma_cross_testing"
    assert window_size == 40
    assert angle_range == (-26.6, 26.6)
    assert near_range == pytest.approx((0.11, 0.12))
    assert above_range == pytest.approx((0.09, 0.1))


def test_parse_strategy_name_without_suffix() -> None:
    """``parse_strategy_name`` should return ``None`` when no suffix is given."""

    base_name, window_size, angle_range, near_range, above_range = parse_strategy_name("ema_sma_cross")
    assert base_name == "ema_sma_cross"
    assert window_size is None
    assert angle_range is None
    assert near_range is None
    assert above_range is None


def test_parse_strategy_name_rejects_malformed_suffix() -> None:
    """``parse_strategy_name`` should raise ``ValueError`` for invalid suffixes."""

    with pytest.raises(ValueError, match="Malformed strategy name"):
        parse_strategy_name("ema_sma_cross_with_slope_")
    with pytest.raises(ValueError, match="positive integer"):
        parse_strategy_name("ema_sma_cross_with_slope_0")


def test_evaluate_combined_strategy_uses_window_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function should parse window suffixes and rename signal columns."""

    date_index = pandas.date_range("2020-01-01", periods=2, freq="D")
    price_data_frame = pandas.DataFrame(
        {"Date": date_index, "open": [10.0, 10.0], "close": [10.0, 10.0]}
    )
    csv_path = tmp_path / "suffix.csv"
    price_data_frame.to_csv(csv_path, index=False)

    captured_arguments: dict[str, int | None] = {"window_size": None}

    def fake_attach_signals(frame: pandas.DataFrame, window_size: int = 50) -> None:
        captured_arguments["window_size"] = window_size
        frame["ema_sma_cross_with_slope_entry_signal"] = [True, False]
        frame["ema_sma_cross_with_slope_exit_signal"] = [False, True]

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        passed_frame: pandas.DataFrame = kwargs["data"]
        assert "ema_sma_cross_with_slope_40_entry_signal" in passed_frame.columns
        assert "ema_sma_cross_with_slope_40_exit_signal" in passed_frame.columns
        trade = Trade(
            entry_date=date_index[0],
            exit_date=date_index[1],
            entry_price=10.0,
            exit_price=11.0,
            profit=1.0,
            holding_period=1,
        )
        return SimulationResult(trades=[trade], total_profit=1.0)

    monkeypatch.setattr(
        strategy, "attach_ema_sma_cross_with_slope_signals", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.BUY_STRATEGIES, "ema_sma_cross_with_slope", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.SELL_STRATEGIES, "ema_sma_cross_with_slope", fake_attach_signals
    )
    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)

    result = evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross_with_slope_40",
        "ema_sma_cross_with_slope_40",
    )

    assert captured_arguments["window_size"] == 40
    assert result.total_trades == 1


def test_evaluate_combined_strategy_passes_near_and_above_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evaluate combined strategy should forward chip thresholds to attach function."""

    date_index = pandas.date_range("2020-01-01", periods=2, freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": [10.0, 10.0],
            "close": [10.0, 10.0],
            "volume": [1.0, 1.0],
        }
    )
    csv_path = tmp_path / "thresholds.csv"
    price_data_frame.to_csv(csv_path, index=False)

    captured_arguments: dict[str, tuple[float, float] | None] = {
        "near_range": None,
        "above_range": None,
    }

    def fake_attach_signals(
        frame: pandas.DataFrame,
        window_size: int = 40,
        angle_range: tuple[float, float] = (-26.6, 26.6),
        near_range: tuple[float, float] = (0.0, 0.12),
        above_range: tuple[float, float] = (0.0, 0.1),
    ) -> None:
        captured_arguments["near_range"] = near_range
        captured_arguments["above_range"] = above_range
        frame["ema_sma_cross_testing_entry_signal"] = [True, False]
        frame["ema_sma_cross_testing_exit_signal"] = [False, True]

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        passed_frame: pandas.DataFrame = kwargs["data"]
        assert (
            "ema_sma_cross_testing_40_-26.6_26.6_0.11,0.12_0.08,0.09_entry_signal"
            in passed_frame.columns
        )
        assert (
            "ema_sma_cross_testing_40_-26.6_26.6_0.11,0.12_0.08,0.09_exit_signal"
            in passed_frame.columns
        )
        trade = Trade(
            entry_date=date_index[0],
            exit_date=date_index[1],
            entry_price=10.0,
            exit_price=11.0,
            profit=1.0,
            holding_period=1,
        )
        return SimulationResult(trades=[trade], total_profit=1.0)

    monkeypatch.setattr(
        strategy, "attach_ema_sma_cross_testing_signals", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.BUY_STRATEGIES, "ema_sma_cross_testing", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.SELL_STRATEGIES, "ema_sma_cross_testing", fake_attach_signals
    )
    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)

    result = evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross_testing_40_-26.6_26.6_0.11,0.12_0.08,0.09",
        "ema_sma_cross_testing_40_-26.6_26.6_0.11,0.12_0.08,0.09",
    )

    assert captured_arguments["near_range"] == pytest.approx((0.11, 0.12))
    assert captured_arguments["above_range"] == pytest.approx((0.08, 0.09))
    assert result.total_trades == 1


def test_compute_signals_for_date_returns_same_day_signal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """compute_signals_for_date should return raw signals on the same day."""

    price_lines = ["Date,open,close,volume\n"]
    start_day = datetime.date(2024, 1, 1)
    for day_index in range(51):
        current_day = start_day + datetime.timedelta(days=day_index)
        price_value = 1.0 if day_index < 50 else 2.0
        price_lines.append(
            f"{current_day.isoformat()},{price_value},{price_value},1000000\n"
        )
    (tmp_path / "AAA.csv").write_text("".join(price_lines), encoding="utf-8")

    monkeypatch.setattr(strategy, "load_symbols_excluded_by_industry", lambda: set())
    monkeypatch.setattr(strategy, "load_ff12_groups_by_symbol", lambda: {})

    same_day_result = strategy.compute_signals_for_date(
        data_directory=tmp_path,
        evaluation_date=pandas.Timestamp("2024-02-20"),
        buy_strategy_name="20_50_sma_cross",
        sell_strategy_name="20_50_sma_cross",
        use_unshifted_signals=True,
    )

    assert same_day_result["entry_signals"] == ["AAA"]
    assert same_day_result["exit_signals"] == []

    shifted_result = strategy.compute_signals_for_date(
        data_directory=tmp_path,
        evaluation_date=pandas.Timestamp("2024-02-20"),
        buy_strategy_name="20_50_sma_cross",
        sell_strategy_name="20_50_sma_cross",
        use_unshifted_signals=False,
    )

    assert shifted_result["entry_signals"] == []


def test_compute_signals_for_date_returns_filtered_symbols_with_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """compute_signals_for_date should include filtered symbols and groups."""

    price_lines = ["Date,open,close,volume\n"]
    start_day = datetime.date(2024, 1, 1)
    for day_index in range(60):
        current_day = start_day + datetime.timedelta(days=day_index)
        price_lines.append(
            f"{current_day.isoformat()},1,1,1000000\n"
        )
    (tmp_path / "AAA.csv").write_text("".join(price_lines), encoding="utf-8")
    (tmp_path / "BBB.csv").write_text("".join(price_lines), encoding="utf-8")

    monkeypatch.setattr(strategy, "load_symbols_excluded_by_industry", lambda: set())
    monkeypatch.setattr(
        strategy,
        "load_ff12_groups_by_symbol",
        lambda: {"AAA": 1, "BBB": 2},
    )

    result = strategy.compute_signals_for_date(
        data_directory=tmp_path,
        evaluation_date=pandas.Timestamp("2024-02-20"),
        buy_strategy_name="20_50_sma_cross",
        sell_strategy_name="20_50_sma_cross",
        top_dollar_volume_rank=2,
        use_unshifted_signals=True,
    )

    assert ("AAA", 1) in result["filtered_symbols"]
    assert ("BBB", 2) in result["filtered_symbols"]


def test_compute_signals_for_date_orders_filtered_symbols_by_dollar_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Filtered symbols should be sorted by descending 50-day dollar volume."""

    start_day = datetime.date(2024, 1, 1)
    total_days = 60
    symbol_volume_pairs = [
        ("HIGH", 3_000_000),
        ("MEDIUM", 2_000_000),
        ("LOW", 1_000_000),
    ]
    for symbol_name, volume_value in symbol_volume_pairs:
        price_lines = ["Date,open,close,volume\n"]
        for day_offset in range(total_days):
            current_day = start_day + datetime.timedelta(days=day_offset)
            price_lines.append(
                f"{current_day.isoformat()},1,1,{volume_value}\n"
            )
        (tmp_path / f"{symbol_name}.csv").write_text(
            "".join(price_lines),
            encoding="utf-8",
        )

    monkeypatch.setattr(strategy, "load_symbols_excluded_by_industry", lambda: set())
    monkeypatch.setattr(
        strategy,
        "load_ff12_groups_by_symbol",
        lambda: {"HIGH": 1, "MEDIUM": 2, "LOW": 3},
    )

    evaluation_day = start_day + datetime.timedelta(days=total_days - 1)
    result = strategy.compute_signals_for_date(
        data_directory=tmp_path,
        evaluation_date=pandas.Timestamp(evaluation_day),
        buy_strategy_name="20_50_sma_cross",
        sell_strategy_name="20_50_sma_cross",
        use_unshifted_signals=True,
    )

    filtered_symbol_names = [symbol_name for symbol_name, _ in result["filtered_symbols"]]
    assert filtered_symbol_names == ["HIGH", "MEDIUM", "LOW"]


def test_calculate_chip_concentration_metrics_defaults_volume_profile_to_none() -> None:
    """Volume profile metrics should be ``None`` when not requested."""

    date_index = pandas.date_range("2020-01-01", periods=60, freq="D")
    close_values = [float(value) for value in range(60, 120)]
    high_values = close_values
    low_values = [value - 1.0 for value in close_values]
    volume_values = [100 for _ in range(60)]
    ohlcv = pandas.DataFrame(
        {
            "Date": date_index,
            "open": close_values,
            "high": high_values,
            "low": low_values,
            "close": close_values,
            "volume": volume_values,
        }
    )

    metrics = calculate_chip_concentration_metrics(ohlcv, lookback_window_size=60)

    assert metrics["hhi"] is None
    assert metrics["distance_to_poc"] is None
    assert metrics["above_volume_ratio_vp"] is None
    assert metrics["below_volume_ratio_vp"] is None
    assert metrics["hvn_count"] is None
    assert metrics["lvn_depth"] is None
