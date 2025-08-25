"""Utility functions for analyzing dollar volume across symbols."""
# TODO: review

from __future__ import annotations

from pathlib import Path

import pandas

from .strategy import load_price_data


def count_symbols_with_average_dollar_volume_above(
    data_directory: Path, minimum_average_dollar_volume: float
) -> int:
    """Return the number of symbols whose 50-day average dollar volume exceeds a threshold.

    Parameters
    ----------
    data_directory : Path
        Directory containing CSV price data for individual symbols.
    minimum_average_dollar_volume : float
        Minimum 50-day average dollar volume, in millions, required for a symbol
        to be counted.

    Returns
    -------
    int
        Number of symbols whose 50-day average dollar volume is greater than
        ``minimum_average_dollar_volume``.

    Raises
    ------
    ValueError
        If a CSV file lacks the ``volume`` column required for dollar volume
        calculations.
    """
    symbol_count = 0
    for csv_file_path in data_directory.glob("*.csv"):
        price_data_frame = load_price_data(csv_file_path)
        if "volume" not in price_data_frame.columns:
            raise ValueError(
                "Volume column is required to compute dollar volume filter"
            )
        dollar_volume_series = (
            price_data_frame["close"] * price_data_frame["volume"]
        )
        if dollar_volume_series.empty:
            continue
        recent_average_dollar_volume = (
            dollar_volume_series.rolling(window=50).mean().iloc[-1] / 1_000_000
        )
        if pandas.isna(recent_average_dollar_volume):
            continue
        if recent_average_dollar_volume > minimum_average_dollar_volume:
            symbol_count += 1
    return symbol_count
