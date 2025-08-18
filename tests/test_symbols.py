"""Tests for the fetch_us_symbols utility."""
# TODO: review

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from stock_indicator.symbols import fetch_us_symbols


def test_fetch_us_symbols_parses_symbols(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The function should return a list of symbols parsed from CSV data."""
    csv_text = "Symbol,Name\nAAPL,Apple Inc.\nMSFT,Microsoft Corp."

    class DummyResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, timeout: int) -> DummyResponse:
        return DummyResponse(csv_text)

    monkeypatch.setattr("stock_indicator.symbols.requests.get", fake_get)
    cache_path = tmp_path / "symbols.json"
    monkeypatch.setattr("stock_indicator.symbols.SYMBOLS_CACHE_PATH", cache_path)

    symbol_list = fetch_us_symbols()
    assert "AAPL" in symbol_list
    assert "MSFT" in symbol_list
    assert cache_path.exists()
    with cache_path.open("r", encoding="utf-8") as cache_file:
        cached_list = json.load(cache_file)
    assert cached_list == symbol_list
