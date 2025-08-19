"""Tests for symbol cache utilities."""
# TODO: review

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from stock_indicator.symbols import load_symbols


def test_load_symbols_fetches_and_reads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The loader should retrieve symbols and cache them locally."""

    symbol_text = "AAA\nBBB"

    class DummyResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    def fake_get(request_url: str, request_timeout: int) -> DummyResponse:  # noqa: ARG001
        return DummyResponse(symbol_text)

    monkeypatch.setattr("stock_indicator.symbols.requests.get", fake_get)
    cache_path = tmp_path / "symbols.txt"
    monkeypatch.setattr("stock_indicator.symbols.SYMBOL_CACHE_PATH", cache_path)

    symbol_list = load_symbols()
    assert "AAA" in symbol_list
    assert "BBB" in symbol_list
    assert cache_path.exists()


def test_load_symbols_reads_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The loader should handle a JSON list of symbols."""

    symbol_sequence = ["AAA", "BBB"]

    class DummyResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    def fake_get(request_url: str, request_timeout: int) -> DummyResponse:  # noqa: ARG001
        return DummyResponse(json.dumps(symbol_sequence))

    monkeypatch.setattr("stock_indicator.symbols.requests.get", fake_get)
    cache_path = tmp_path / "symbols.txt"
    monkeypatch.setattr("stock_indicator.symbols.SYMBOL_CACHE_PATH", cache_path)

    symbol_list = load_symbols()
    assert symbol_list == symbol_sequence
    assert cache_path.exists()
