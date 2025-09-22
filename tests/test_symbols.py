"""Tests for symbol cache utilities."""
# TODO: review

import os
import sys
from pathlib import Path

import pandas
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)


def test_load_symbols_fetches_and_caches_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The loader should retrieve symbols, cache them, and return the parsed list."""

    import stock_indicator.symbols as symbols_module

    cache_path = tmp_path / "symbols.txt"
    monkeypatch.setattr(symbols_module, "SYMBOL_CACHE_PATH", cache_path)

    call_counter = {"count": 0}

    def fake_load_company_tickers() -> pandas.DataFrame:
        call_counter["count"] += 1
        return pandas.DataFrame({"ticker": ["AAA", "bbb", None, "AAA"]})

    monkeypatch.setattr(
        "stock_indicator.sector_pipeline.sec_api.load_company_tickers",
        fake_load_company_tickers,
    )

    symbol_list = symbols_module.load_symbols()
    assert symbol_list == ["AAA", "BBB"]
    assert cache_path.exists()

    symbol_list_second = symbols_module.load_symbols()
    assert symbol_list_second == ["AAA", "BBB"]
    assert call_counter["count"] == 1
