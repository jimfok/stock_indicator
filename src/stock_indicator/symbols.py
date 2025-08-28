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
    """Build a list of symbols that have data available from Yahoo Finance.

    This function reads the SEC-derived cache (``symbols.txt``), probes Yahoo
    Finance for a very short period for each symbol, and writes the subset that
    returns non-empty data to ``symbols_yf.txt`` as a newline-separated list.

    Returns the list of symbols written.
    """
    # Local import to avoid imposing yfinance as a hard dependency for callers
    import yfinance  # type: ignore

    base_symbol_list = load_symbols()
    available_list: List[str] = []
    for ticker in base_symbol_list:
        if ticker == SP500_SYMBOL:
            # Exclude the index symbol from the YF company universe list
            continue
        try:
            frame = yfinance.download(
                ticker,
                period="5d",
                progress=False,
                auto_adjust=True,
            )
        except Exception as error:  # noqa: BLE001
            LOGGER.debug("Probe failed for %s: %s", ticker, error)
            continue
        if not frame.empty:
            available_list.append(ticker)
    available_list = sorted(set(available_list))
    YF_SYMBOL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    YF_SYMBOL_CACHE_PATH.write_text("\n".join(available_list) + "\n", encoding="utf-8")
    LOGGER.info("YF symbol cache written to %s (count=%d)", YF_SYMBOL_CACHE_PATH, len(available_list))
    return available_list


def load_yf_symbols() -> list[str]:
    """Return the list of symbols confirmed to have Yahoo Finance data.

    Reads ``symbols_yf.txt`` as a newline-separated list. If the file does not
    exist, an empty list is returned.
    """
    if not YF_SYMBOL_CACHE_PATH.exists():
        return []
    file_content = YF_SYMBOL_CACHE_PATH.read_text(encoding="utf-8")
    return [line.strip() for line in file_content.splitlines() if line.strip()]
