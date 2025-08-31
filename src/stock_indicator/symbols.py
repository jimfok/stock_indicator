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
YF_SYMBOL_CACHE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "symbols_yf.txt"
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


def update_yf_symbol_cache() -> List[str]:
    """Rebuild the YF-ready symbol list by scanning local CSVs.

    This treats ``data/symbols_yf.txt`` as the set of symbols that are ready
    for simulations and daily updates. The function scans
    ``data/stock_data/*.csv`` and writes the corresponding symbols (file names
    without ``.csv``) to ``symbols_yf.txt``. The S&P 500 index symbol is
    excluded.

    Returns the list of symbols written.
    """
    data_directory = Path(__file__).resolve().parent.parent.parent / "data"
    stock_data_directory = data_directory / "stock_data"
    discovered_symbols: List[str] = []
    if stock_data_directory.exists():
        for csv_path in stock_data_directory.glob("*.csv"):
            symbol_name = csv_path.stem.strip().upper()
            if not symbol_name or symbol_name == SP500_SYMBOL:
                continue
            discovered_symbols.append(symbol_name)
    discovered_symbols = sorted(set(discovered_symbols))
    YF_SYMBOL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    YF_SYMBOL_CACHE_PATH.write_text("\n".join(discovered_symbols) + "\n", encoding="utf-8")
    LOGGER.info(
        "YF symbol cache rebuilt from %s (count=%d)",
        stock_data_directory,
        len(discovered_symbols),
    )
    return discovered_symbols


def load_yf_symbols() -> list[str]:
    """Return the list of symbols confirmed to have Yahoo Finance data.

    Reads ``symbols_yf.txt`` as a newline-separated list. If the file does not
    exist, an empty list is returned.
    """
    if not YF_SYMBOL_CACHE_PATH.exists():
        return []
    file_content = YF_SYMBOL_CACHE_PATH.read_text(encoding="utf-8")
    return [line.strip() for line in file_content.splitlines() if line.strip()]


def add_symbol_to_yf_cache(symbol: str) -> bool:
    """Add ``symbol`` to the Yahoo Finance symbol cache file.

    Ensures the cache at ``data/symbols_yf.txt`` exists and contains the
    upper-cased ``symbol`` exactly once. The S&P 500 index symbol
    (``SP500_SYMBOL``) is never added. Returns ``True`` when the list was
    modified, ``False`` otherwise.

    Parameters
    ----------
    symbol: str
        Ticker symbol to add to the YF cache.
    """
    normalized_symbol = (symbol or "").strip().upper()
    if not normalized_symbol or normalized_symbol == SP500_SYMBOL:
        return False
    existing_symbols = set(load_yf_symbols())
    if normalized_symbol in existing_symbols:
        return False
    updated_symbols = sorted(existing_symbols | {normalized_symbol})
    YF_SYMBOL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    YF_SYMBOL_CACHE_PATH.write_text("\n".join(updated_symbols) + "\n", encoding="utf-8")
    LOGGER.info("Added %s to YF symbol cache (%s)", normalized_symbol, YF_SYMBOL_CACHE_PATH)
    return True
