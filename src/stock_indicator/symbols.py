"""Utility functions for retrieving stock symbols."""
# TODO: review

from __future__ import annotations

import csv
import json
from pathlib import Path

import requests

SYMBOLS_URL = (
    "https://raw.githubusercontent.com/datasets/"
    "us-stock-symbols/master/data/nyse-listed.csv"
)
SYMBOLS_CACHE_PATH = Path(__file__).with_name("us_symbols.json")


def fetch_us_symbols() -> list[str]:
    """Fetch a list of U.S. stock ticker symbols.

    The function downloads symbol data from a public GitHub dataset and caches
    it locally to avoid repeated network requests.

    Returns
    -------
    list[str]
        List of stock ticker symbols.

    Raises
    ------
    requests.RequestException
        If the remote request fails.
    """
    if SYMBOLS_CACHE_PATH.exists():
        with SYMBOLS_CACHE_PATH.open("r", encoding="utf-8") as cache_file:
            return json.load(cache_file)

    response = requests.get(SYMBOLS_URL, timeout=30)
    response.raise_for_status()
    csv_text = response.text
    csv_reader = csv.DictReader(csv_text.splitlines())
    symbol_list = [row["Symbol"].strip() for row in csv_reader if row.get("Symbol")]
    with SYMBOLS_CACHE_PATH.open("w", encoding="utf-8") as cache_file:
        json.dump(symbol_list, cache_file)
    return symbol_list
