"""Functions for downloading historical stock market data."""
# TODO: review

from __future__ import annotations

import logging
import time

import pandas
import yfinance

LOGGER = logging.getLogger(__name__)


def download_history(symbol: str, start: str, end: str) -> pandas.DataFrame:
    """Download historical price data for a stock symbol.

    Parameters
    ----------
    symbol: str
        Stock ticker symbol to download.
    start: str
        Start date in ISO format (``YYYY-MM-DD``).
    end: str
        End date in ISO format (``YYYY-MM-DD``).

    Returns
    -------
    pandas.DataFrame
        Data frame containing the historical data.

    Raises
    ------
    Exception
        Propagates the last error if downloading repeatedly fails.
    """
    maximum_attempts = 3
    for attempt_number in range(1, maximum_attempts + 1):
        try:
            downloaded_frame = yfinance.download(
                symbol,
                start=start,
                end=end,
                progress=False,
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
