"""Tests for symbol cache utilities."""
# TODO: review

import json
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)


def test_load_symbols_fetches_and_caches_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The loader should retrieve symbols, cache them, and return the parsed list."""

    mock_symbol_list = ["AAA", "BBB"]
    json_text = json.dumps(mock_symbol_list)

    class DummyResponse:
        """Simple container for mocked text responses."""

        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    request_call_count = {"count": 0}

    def fake_get(request_url: str, timeout: int) -> DummyResponse:  # noqa: ARG001
        """Return a dummy response and track request invocations."""

        request_call_count["count"] += 1
        return DummyResponse(json_text)

    requests_stub = types.ModuleType("requests")
    requests_stub.get = fake_get
    monkeypatch.setitem(sys.modules, "requests", requests_stub)

    import stock_indicator.symbols as symbols_module
    cache_path = tmp_path / "symbols.txt"
    monkeypatch.setattr(symbols_module, "SYMBOL_CACHE_PATH", cache_path)

    symbol_list = symbols_module.load_symbols()
    expected_list = mock_symbol_list + [symbols_module.SP500_SYMBOL]
    assert symbol_list == expected_list
    assert cache_path.exists()

    symbol_list_second = symbols_module.load_symbols()
    assert symbol_list_second == expected_list
    assert request_call_count["count"] == 1


def test_reset_daily_job_symbols_copies_from_yahoo_finance_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reset should copy symbols from the Yahoo Finance cache."""

    import stock_indicator.symbols as symbols_module

    daily_job_file_path = tmp_path / "symbols_daily_job.txt"
    yahoo_finance_file_path = tmp_path / "symbols_yf.txt"
    yahoo_finance_file_path.write_text("AAA\nBBB\n", encoding="utf-8")

    monkeypatch.setattr(symbols_module, "DAILY_JOB_SYMBOLS_PATH", daily_job_file_path)
    monkeypatch.setattr(symbols_module, "YF_SYMBOL_CACHE_PATH", yahoo_finance_file_path)

    written_symbols = symbols_module.reset_daily_job_symbols()
    assert written_symbols == ["AAA", "BBB"]
    assert daily_job_file_path.read_text(encoding="utf-8") == "AAA\nBBB\n"


def test_load_daily_job_symbols_creates_file_if_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Loading should initialize the daily job file when it is absent."""

    import stock_indicator.symbols as symbols_module

    daily_job_file_path = tmp_path / "symbols_daily_job.txt"
    yahoo_finance_file_path = tmp_path / "symbols_yf.txt"
    yahoo_finance_file_path.write_text("AAA\nBBB\n", encoding="utf-8")

    monkeypatch.setattr(symbols_module, "DAILY_JOB_SYMBOLS_PATH", daily_job_file_path)
    monkeypatch.setattr(symbols_module, "YF_SYMBOL_CACHE_PATH", yahoo_finance_file_path)

    loaded_symbols = symbols_module.load_daily_job_symbols()
    assert loaded_symbols == ["AAA", "BBB"]
    assert daily_job_file_path.read_text(encoding="utf-8") == "AAA\nBBB\n"


def test_remove_daily_job_symbol_updates_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Removing a symbol should update the daily job list."""

    import stock_indicator.symbols as symbols_module

    daily_job_file_path = tmp_path / "symbols_daily_job.txt"
    yahoo_finance_file_path = tmp_path / "symbols_yf.txt"
    yahoo_finance_file_path.write_text("AAA\nBBB\nCCC\n", encoding="utf-8")
    daily_job_file_path.write_text("AAA\nBBB\nCCC\n", encoding="utf-8")

    monkeypatch.setattr(symbols_module, "DAILY_JOB_SYMBOLS_PATH", daily_job_file_path)
    monkeypatch.setattr(symbols_module, "YF_SYMBOL_CACHE_PATH", yahoo_finance_file_path)

    was_removed = symbols_module.remove_daily_job_symbol("bbb")
    assert was_removed is True
    assert daily_job_file_path.read_text(encoding="utf-8") == "AAA\nCCC\n"

    remaining_symbols = symbols_module.load_daily_job_symbols()
    assert remaining_symbols == ["AAA", "CCC"]

    was_removed_again = symbols_module.remove_daily_job_symbol("BBB")
    assert was_removed_again is False

