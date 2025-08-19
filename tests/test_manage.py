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
        return pandas.DataFrame({"close": [1.0]})

    monkeypatch.setattr(
        manage_module.data_loader, "download_history", fake_download_history
    )
    monkeypatch.setattr(manage_module, "DATA_DIRECTORY", tmp_path)

    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd("update_data TEST 2023-01-01 2023-01-02")
    output_file = tmp_path / "TEST.csv"
    assert output_file.exists()
    assert recorded_arguments == {
        "symbol": "TEST",
        "start": "2023-01-01",
        "end": "2023-01-02",
    }


def test_update_all_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The command should download data for every symbol in the cache."""
    import stock_indicator.manage as manage_module

    symbol_list = ["AAA", "BBB"]

    def fake_load_symbols() -> list[str]:
        return symbol_list

    download_calls: list[str] = []

    def fake_download_history(symbol: str, start: str, end: str) -> pandas.DataFrame:
        download_calls.append(symbol)
        return pandas.DataFrame({"close": [1.0]})

    monkeypatch.setattr(manage_module.symbols, "load_symbols", fake_load_symbols)
    monkeypatch.setattr(
        manage_module.data_loader, "download_history", fake_download_history
    )
    monkeypatch.setattr(manage_module, "DATA_DIRECTORY", tmp_path)

    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd("update_all_data 2023-01-01 2023-01-02")

    for symbol in symbol_list:
        assert (tmp_path / f"{symbol}.csv").exists()
    assert download_calls == symbol_list
