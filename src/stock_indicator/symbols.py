"""Utilities for maintaining a local cache of stock symbols."""
# TODO: review

from __future__ import annotations

import json
import logging
from pathlib import Path

import requests

LOGGER = logging.getLogger(__name__)

# Raw text file with one ticker symbol per line.
SYMBOL_SOURCE_TEXT_URL = (
    "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
)
SYMBOL_CACHE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "symbols.txt"
)


def update_symbol_cache() -> None:
    """Download the latest symbol list and store it locally.

    The remote endpoint may return either a newline separated text file or a JSON
    encoded list of ticker symbols. The content is saved verbatim and parsed when
    loaded.
    """
    response = requests.get(SYMBOL_SOURCE_TEXT_URL, timeout=30)
    response.raise_for_status()
    SYMBOL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYMBOL_CACHE_PATH.write_text(response.text, encoding="utf-8")
    LOGGER.info("Symbol cache written to %s", SYMBOL_CACHE_PATH)


def load_symbols() -> list[str]:
    """Return the list of symbols from the local cache.

    The cache file may contain one ticker per line or a JSON encoded list of
    strings representing ticker symbols.
    """
    if not SYMBOL_CACHE_PATH.exists():
        update_symbol_cache()
    file_content = SYMBOL_CACHE_PATH.read_text(encoding="utf-8")
    try:
        parsed_symbols = json.loads(file_content)
    except json.JSONDecodeError:
        symbol_list = [
            line.strip()
            for line in file_content.splitlines()
            if line.strip()
        ]
    else:
        if not isinstance(parsed_symbols, list) or not all(
            isinstance(symbol, str) for symbol in parsed_symbols
        ):
            raise ValueError("Symbol cache JSON must be a list of strings.")
        symbol_list = parsed_symbols
    return symbol_list

