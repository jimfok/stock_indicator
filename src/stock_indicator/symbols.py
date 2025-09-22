"""Utilities for maintaining a local cache of stock symbols.

This module now builds the local symbol cache from the SEC company tickers
dataset used by the sector classification pipeline, instead of downloading a
third-party aggregate list. The cache is stored as a newline-separated text
file at ``data/symbols.txt``.
"""
# TODO: review

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

LOGGER = logging.getLogger(__name__)

SYMBOL_CACHE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "symbols.txt"
)

# Symbol representing the S&P 500 index.
SP500_SYMBOL = "^GSPC"


def update_symbol_cache() -> None:
    """Build the local symbol cache from the SEC company tickers dataset.

    Uses the sector pipeline's SEC integration to fetch the authoritative list
    of company tickers and writes them as a newline-separated list to
    ``data/symbols.txt``.
    """
    from stock_indicator.sector_pipeline.sec_api import load_company_tickers

    company_table = load_company_tickers()
    # Ensure unique, normalized, and deterministically ordered symbols
    tickers: List[str] = (
        company_table["ticker"].dropna().astype(str).str.upper().drop_duplicates().sort_values().tolist()
    )
    SYMBOL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYMBOL_CACHE_PATH.write_text("\n".join(tickers) + "\n", encoding="utf-8")
    LOGGER.info("Symbol cache written to %s (from SEC company_tickers.json)", SYMBOL_CACHE_PATH)


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


