# TODO: review

import logging
import os
import sys
from typing import Callable

import pandas
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from stock_indicator import cli
from stock_indicator.simulator import SimulationResult


def test_parser_handles_arguments() -> None:
    """The parser should correctly read required arguments."""
    parser = cli.create_parser()
    parsed_arguments = parser.parse_args(
        [
            "--symbol",
            "AAA",
            "--start",
            "2022-01-01",
            "--end",
            "2022-02-01",
            "--strategy",
            "sma",
        ]
    )
    assert parsed_arguments.symbol == "AAA"
    assert parsed_arguments.start == "2022-01-01"
    assert parsed_arguments.end == "2022-02-01"
    assert parsed_arguments.strategy == "sma"
    assert parsed_arguments.price_column == "adj_close"


def test_run_cli_invokes_components(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The CLI should call underlying modules and log the result."""

    def fake_load_symbols() -> list[str]:
        return ["AAA"]

    def fake_download_history(symbol: str, start: str, end: str) -> pandas.DataFrame:
        assert symbol == "AAA"
        assert start == "2022-01-01"
        assert end == "2022-02-01"
        return pandas.DataFrame({"Adj Close": [1.0, 2.0, 3.0]})

    def fake_sma(price_series: pandas.Series, window_size: int) -> pandas.Series:
        return pandas.Series([1.0, 1.5, 2.0])

    def fake_simulate_trades(
        data_frame: pandas.DataFrame,
        entry_rule: Callable[[pandas.Series], bool],
        exit_rule: Callable[[pandas.Series, pandas.Series], bool],
    ) -> SimulationResult:
        return SimulationResult(trades=[], total_profit=5.0)

    monkeypatch.setattr(cli.symbols, "load_symbols", fake_load_symbols)
    monkeypatch.setattr(cli.data_loader, "download_history", fake_download_history)
    monkeypatch.setattr(cli.indicators, "sma", fake_sma)
    monkeypatch.setattr(cli.simulator, "simulate_trades", fake_simulate_trades)

    with caplog.at_level(logging.INFO):
        cli.run_cli(
            [
                "--symbol",
                "AAA",
                "--start",
                "2022-01-01",
                "--end",
                "2022-02-01",
                "--strategy",
                "sma",
            ]
        )
    assert "Total profit: 5.0" in caplog.text


def test_run_cli_uses_adj_close_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI should use the ``adj_close`` column when no option is given."""

    def fake_load_symbols() -> list[str]:
        return ["AAA"]

    def fake_download_history(symbol: str, start: str, end: str) -> pandas.DataFrame:
        assert symbol == "AAA"
        assert start == "2022-01-01"
        assert end == "2022-02-01"
        return pandas.DataFrame(
            {"Adj Close": [2.0, 2.0, 2.0], "Close": [-1.0, -1.0, -1.0]}
        )

    def fake_sma(price_series: pandas.Series, window_size: int) -> pandas.Series:
        assert price_series.equals(
            pandas.Series([2.0, 2.0, 2.0], name="adj_close")
        )
        return pandas.Series([0.0, 0.0, 0.0])

    def fake_simulate_trades(
        data_frame: pandas.DataFrame,
        entry_rule: Callable[[pandas.Series], bool],
        exit_rule: Callable[[pandas.Series, pandas.Series], bool],
    ) -> SimulationResult:
        assert entry_rule(data_frame.iloc[0])
        assert not exit_rule(data_frame.iloc[0], data_frame.iloc[0])
        return SimulationResult(trades=[], total_profit=0.0)

    monkeypatch.setattr(cli.symbols, "load_symbols", fake_load_symbols)
    monkeypatch.setattr(cli.data_loader, "download_history", fake_download_history)
    monkeypatch.setattr(cli.indicators, "sma", fake_sma)
    monkeypatch.setattr(cli.simulator, "simulate_trades", fake_simulate_trades)

    cli.run_cli(
        [
            "--symbol",
            "AAA",
            "--start",
            "2022-01-01",
            "--end",
            "2022-02-01",
            "--strategy",
            "sma",
        ]
    )


def test_run_cli_accepts_price_column_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI should use the specified column when ``--price-column`` is set."""

    def fake_load_symbols() -> list[str]:
        return ["AAA"]

    def fake_download_history(symbol: str, start: str, end: str) -> pandas.DataFrame:
        assert symbol == "AAA"
        assert start == "2022-01-01"
        assert end == "2022-02-01"
        return pandas.DataFrame(
            {"Adj Close": [2.0, 2.0, 2.0], "Close": [-1.0, -1.0, -1.0]}
        )

    def fake_sma(price_series: pandas.Series, window_size: int) -> pandas.Series:
        assert price_series.equals(
            pandas.Series([-1.0, -1.0, -1.0], name="close")
        )
        return pandas.Series([0.0, 0.0, 0.0])

    def fake_simulate_trades(
        data_frame: pandas.DataFrame,
        entry_rule: Callable[[pandas.Series], bool],
        exit_rule: Callable[[pandas.Series, pandas.Series], bool],
    ) -> SimulationResult:
        assert not entry_rule(data_frame.iloc[0])
        assert exit_rule(data_frame.iloc[0], data_frame.iloc[0])
        return SimulationResult(trades=[], total_profit=0.0)

    monkeypatch.setattr(cli.symbols, "load_symbols", fake_load_symbols)
    monkeypatch.setattr(cli.data_loader, "download_history", fake_download_history)
    monkeypatch.setattr(cli.indicators, "sma", fake_sma)
    monkeypatch.setattr(cli.simulator, "simulate_trades", fake_simulate_trades)

    cli.run_cli(
        [
            "--symbol",
            "AAA",
            "--start",
            "2022-01-01",
            "--end",
            "2022-02-01",
            "--strategy",
            "sma",
            "--price-column",
            "close",
        ]
    )
