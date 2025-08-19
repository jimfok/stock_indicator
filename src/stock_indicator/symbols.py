"""Utilities for maintaining a local cache of stock symbols."""
# TODO: review

from __future__ import annotations

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
    """Download the latest symbol list and store it locally."""
    response = requests.get(SYMBOL_SOURCE_TEXT_URL, timeout=30)
    response.raise_for_status()
    SYMBOL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYMBOL_CACHE_PATH.write_text(response.text, encoding="utf-8")
    LOGGER.info("Symbol cache written to %s", SYMBOL_CACHE_PATH)


def load_symbols() -> list[str]:
    """Return the list of symbols from the local cache.

    The cache file contains one ticker symbol per line.
    """
    if not SYMBOL_CACHE_PATH.exists():
        update_symbol_cache()
    with SYMBOL_CACHE_PATH.open("r", encoding="utf-8") as symbol_file:
        symbol_list: list[str] = []
        for symbol_line in symbol_file:
            stripped_symbol = symbol_line.strip()
            if stripped_symbol:
                symbol_list.append(stripped_symbol)
    return symbol_list

