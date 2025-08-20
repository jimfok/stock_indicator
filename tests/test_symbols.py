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

