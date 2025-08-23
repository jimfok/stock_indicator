"""Tests for the interactive management shell."""

# TODO: review

import io
import os
import sys
from pathlib import Path

import pandas
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)


def test_update_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command should invoke the symbol cache update."""
    import stock_indicator.manage as manage_module

    call_record = {"called": False}

    def fake_update_symbol_cache() -> None:
        call_record["called"] = True

    monkeypatch.setattr(
        manage_module.symbols, "update_symbol_cache", fake_update_symbol_cache
    )

    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd("update_symbols")
    assert call_record["called"] is True


def test_update_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The command should download data and write it to a CSV file."""
    import stock_indicator.manage as manage_module

    recorded_arguments: dict[str, str] = {}

    def fake_download_history(symbol: str, start: str, end: str) -> pandas.DataFrame:
        recorded_arguments["symbol"] = symbol
        recorded_arguments["start"] = start
        recorded_arguments["end"] = end
        return pandas.DataFrame(
            {"close": [1.0]}, index=pandas.to_datetime(["2023-01-01"])
        ).rename_axis("Date")

    monkeypatch.setattr(
        manage_module.data_loader, "download_history", fake_download_history
    )
    monkeypatch.setattr(manage_module, "DATA_DIRECTORY", tmp_path)

    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd("update_data TEST 2023-01-01 2023-01-02")
    output_file = tmp_path / "TEST.csv"
    assert output_file.exists()
    csv_contents = pandas.read_csv(output_file)
    assert "Date" in csv_contents.columns
    assert recorded_arguments == {
        "symbol": "TEST",
        "start": "2023-01-01",
        "end": "2023-01-02",
    }


def test_update_all_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The command should download data for every symbol in the cache."""
    import stock_indicator.manage as manage_module

    symbol_list = ["AAA", "BBB", manage_module.SP500_SYMBOL]

    def fake_load_symbols() -> list[str]:
        return symbol_list

    download_calls: list[str] = []

    def fake_download_history(symbol: str, start: str, end: str) -> pandas.DataFrame:
        download_calls.append(symbol)
        return pandas.DataFrame(
            {"close": [1.0]}, index=pandas.to_datetime(["2023-01-01"])
        ).rename_axis("Date")

    monkeypatch.setattr(manage_module.symbols, "load_symbols", fake_load_symbols)
    monkeypatch.setattr(
        manage_module.data_loader, "download_history", fake_download_history
    )
    monkeypatch.setattr(manage_module, "DATA_DIRECTORY", tmp_path)

    expected_symbols = symbol_list
    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd("update_all_data 2023-01-01 2023-01-02")
    for symbol in expected_symbols:
        csv_path = tmp_path / f"{symbol}.csv"
        assert csv_path.exists()
        csv_contents = pandas.read_csv(csv_path)
        assert "Date" in csv_contents.columns
    assert download_calls == expected_symbols


def test_start_simulate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command should evaluate the EMA/SMA cross strategy."""
    import stock_indicator.manage as manage_module

    call_record: dict[str, bool] = {"called": False}

    from stock_indicator.strategy import StrategyMetrics

    def fake_evaluate(data_directory: Path) -> StrategyMetrics:
        call_record["called"] = True
        assert data_directory == manage_module.DATA_DIRECTORY
        return StrategyMetrics(
            total_trades=3,
            win_rate=0.5,
            mean_profit_percentage=0.1,
            profit_percentage_standard_deviation=0.0,
            mean_loss_percentage=0.05,
            loss_percentage_standard_deviation=0.0,
        )

    monkeypatch.setattr(
        manage_module.strategy,
        "evaluate_ema_sma_cross_strategy",
        fake_evaluate,
    )

    output_buffer = io.StringIO()
    shell = manage_module.StockShell(stdout=output_buffer)
    shell.onecmd("start_simulate ema_sma_cross ema_sma_cross")
    assert call_record["called"] is True
    assert (
        "Trades: 3, Win rate: 50.00%, Mean profit %: 10.00%, Profit % Std Dev: 0.00%, "
        "Mean loss %: 5.00%, Loss % Std Dev: 0.00%" in output_buffer.getvalue()
    )
