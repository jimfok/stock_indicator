"""Tests for the download_history utility."""
# TODO: review

import logging
import os
import sys
import types

import pandas
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Create a minimal yfinance stub before importing the module under test
fake_yfinance_module = types.ModuleType("yfinance")
fake_yfinance_module.download = lambda *args, **kwargs: pandas.DataFrame()
sys.modules["yfinance"] = fake_yfinance_module

from stock_indicator.data_loader import download_history


def test_download_history_returns_dataframe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The function should return data provided by yfinance."""
    raw_dataframe = pandas.DataFrame({"Close": [1.0, 2.0]})

    def stubbed_download(
        symbol: str,
        start: str,
        end: str,
        progress: bool = False,
    ) -> pandas.DataFrame:
        return raw_dataframe

    monkeypatch.setattr(
        "stock_indicator.data_loader.yfinance.download", stubbed_download
    )
    monkeypatch.setattr("stock_indicator.symbols.load_symbols", lambda: ["TEST"])
    result_dataframe = download_history("TEST", "2021-01-01", "2021-01-02")
    expected_dataframe = raw_dataframe.rename(
        columns=lambda name: name.lower().replace(" ", "_")
    )
    pandas.testing.assert_frame_equal(result_dataframe, expected_dataframe)


def test_download_history_retries_on_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The function should retry and log warnings on temporary failures."""
    call_counter = {"count": 0}
    raw_dataframe = pandas.DataFrame({"Close": [1.0]})

    def flaky_download(
        symbol: str,
        start: str,
        end: str,
        progress: bool = False,
    ) -> pandas.DataFrame:
        call_counter["count"] += 1
        if call_counter["count"] < 3:
            raise ValueError("temporary error")
        return raw_dataframe

    monkeypatch.setattr(
        "stock_indicator.data_loader.yfinance.download", flaky_download
    )
    monkeypatch.setattr("stock_indicator.symbols.load_symbols", lambda: ["TEST"])
    with caplog.at_level(logging.WARNING):
        result_dataframe = download_history("TEST", "2021-01-01", "2021-01-02")

    assert call_counter["count"] == 3
    expected_dataframe = raw_dataframe.rename(
        columns=lambda name: name.lower().replace(" ", "_")
    )
    pandas.testing.assert_frame_equal(result_dataframe, expected_dataframe)
    assert "Attempt 1 to download data for TEST failed" in caplog.text


def test_download_history_raises_after_max_attempts(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The function should raise the last error after exhausting retries."""

    def failing_download(
        symbol: str,
        start: str,
        end: str,
        progress: bool = False,
    ) -> pandas.DataFrame:
        raise ValueError("permanent error")

    monkeypatch.setattr(
        "stock_indicator.data_loader.yfinance.download", failing_download
    )
    monkeypatch.setattr("stock_indicator.symbols.load_symbols", lambda: ["TEST"])
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            download_history("TEST", "2021-01-01", "2021-01-02")
    assert "Failed to download data for TEST after" in caplog.text


def test_download_history_forwards_optional_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The function should forward optional keyword arguments to yfinance."""
    captured_arguments: dict[str, str] = {}

    def stubbed_download(
        symbol: str,
        start: str,
        end: str,
        progress: bool = False,
        **options: str,
    ) -> pandas.DataFrame:
        captured_arguments.update(options)
        return pandas.DataFrame()

    monkeypatch.setattr(
        "stock_indicator.data_loader.yfinance.download", stubbed_download
    )
    monkeypatch.setattr("stock_indicator.symbols.load_symbols", lambda: ["TEST"])
    download_history("TEST", "2021-01-01", "2021-01-02", interval="1h")
    assert captured_arguments["interval"] == "1h"
