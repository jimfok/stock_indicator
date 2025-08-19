"""Functions for downloading historical stock market data."""
# TODO: review

from __future__ import annotations

import logging
import time
from typing import Any

import pandas
import yfinance

from .symbols import load_symbols

LOGGER = logging.getLogger(__name__)


def download_history(
    symbol: str,
    start: str,
    end: str,
    **download_options: Any,
) -> pandas.DataFrame:
    """Download historical price data for a stock symbol.

    Parameters
    ----------
    symbol: str
        Stock ticker symbol to download.
    start: str
        Start date in ISO format (``YYYY-MM-DD``).
    end: str
        End date in ISO format (``YYYY-MM-DD``).
    **download_options
        Additional keyword arguments forwarded to :func:`yfinance.download`, such
        as ``actions``, ``auto_adjust``, or ``interval``.

    Returns
    -------
    pandas.DataFrame
        Data frame containing the historical data.

    Raises
    ------
    ValueError
        If the provided symbol is not known.
    Exception
        Propagates the last error if downloading repeatedly fails.
    """
    available_symbol_list = load_symbols()
    if available_symbol_list and symbol not in available_symbol_list:
        raise ValueError(f"Unknown symbol: {symbol}")

    maximum_attempts = 3
    for attempt_number in range(1, maximum_attempts + 1):
        try:
            downloaded_frame = yfinance.download(
                symbol,
                start=start,
                end=end,
                progress=False,
                **download_options,
            )
            return downloaded_frame
        except Exception as download_error:  # noqa: BLE001
            LOGGER.warning(
                "Attempt %d to download data for %s failed: %s",
                attempt_number,
                symbol,
                download_error,
            )
            if attempt_number == maximum_attempts:
                LOGGER.error(
                    "Failed to download data for %s after %d attempts",
                    symbol,
                    maximum_attempts,
                )
                raise
            time.sleep(1)
